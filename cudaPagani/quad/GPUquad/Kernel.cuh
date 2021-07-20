#ifndef CUDACUHRE_QUAD_GPUQUAD_KERNEL_CUH
#define CUDACUHRE_QUAD_GPUQUAD_KERNEL_CUH

#include "cudaPagani/quad/util/Volume.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>

#include "cudaPagani/quad/GPUquad/PartitionManager.cuh"
#include "cudaPagani/quad/GPUquad/Phases.cuh"
#include "cudaPagani/quad/GPUquad/Rule.cuh"

#include "nvToolsExt.h"
#include <cuda.h>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>

//#define startRegions 256 //for 8D
#define startRegions 1 // if starting with 1 region
namespace quad {
  using namespace cooperative_groups;

  //===========
  // FOR DEBUGGINGG
  void
  PrintToFile(std::string outString, std::string filename, bool appendMode = 0)
  {
    if (appendMode) {
      std::ofstream outfile(filename, std::ios::app);
      outfile << outString << std::endl;
      outfile.close();
    } else {
      std::ofstream outfile(filename);
      outfile << outString << std::endl;
      outfile.close();
    }
  }

  //==========
  __constant__ size_t dFEvalPerRegion;

  // delete, the display function does this now

  // delete, now used anymore
  void
  FinalDataPrint(std::stringstream& outfile,
                 std::string id,
                 double true_value,
                 double epsrel,
                 double epsabs,
                 double value,
                 double error,
                 double nregions,
                 double status,
                 int _final,
                 double time,
                 std::string filename,
                 bool appendMode = 0)
  {

    std::ostringstream streamObj;
    std::ostringstream streamObj2;
    streamObj << value;
    streamObj2 << error;

    if (appendMode == 0)
      outfile << "id, value, epsrel, epsabs, estimate, errorest, regions, "
                 "converge, final, total_time"
              << std::endl;
    outfile << std::setprecision(18);
    outfile << id << ",\t" << std::to_string(true_value) << ",\t" << epsrel
            << ",\t" << epsabs << ",\t" << value << ",\t" << error << ",\t"
            << nregions << ",\t" << status << ",\t" << _final << ",\t" << time
            << std::endl;

    // std::cout<<outfile.str()<<std::endl;
    PrintToFile(outfile.str(), filename, appendMode);
  }

  template <typename T>
  __global__ void
  generateInitialRegions(T* dRegions,
                         T* dRegionsLength,
                         size_t numRegions,
                         T* newRegions,
                         T* newRegionsLength,
                         size_t newNumOfRegions,
                         int numOfDivisionsPerRegionPerDimension,
                         int NDIM)
  {

    extern __shared__ T slength[];
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < NDIM) {
      slength[threadIdx.x] =
        dRegionsLength[threadIdx.x] / numOfDivisionsPerRegionPerDimension;
    }
    __syncthreads();

    if (threadId < newNumOfRegions) {
      size_t interval_index =
        threadId / pow((T)numOfDivisionsPerRegionPerDimension, (T)NDIM);
      size_t local_id =
        threadId % (size_t)pow((T)numOfDivisionsPerRegionPerDimension, (T)NDIM);
      for (int dim = 0; dim < NDIM; ++dim) {
        size_t id =
          (size_t)(local_id /
                   pow((T)numOfDivisionsPerRegionPerDimension, (T)dim)) %
          numOfDivisionsPerRegionPerDimension;
        newRegions[newNumOfRegions * dim + threadId] =
          dRegions[numRegions * dim + interval_index] + id * slength[dim];
        newRegionsLength[newNumOfRegions * dim + threadId] = slength[dim];
      }
    }
  }

  template <typename T, int NDIM>
  __global__ void
  alignRegions(T* dRegions,
               T* dRegionsLength,
               int* activeRegions,
               T* dRegionsIntegral,
               T* dRegionsError,
               T* dRegionsParentIntegral,
               T* dRegionsParentError,
               int* subDividingDimension,
               int* scannedArray,
               T* newActiveRegions,
               T* newActiveRegionsLength,
               int* newActiveRegionsBisectDim,
               size_t numRegions,
               size_t newNumRegions,
               int numOfDivisionOnDimension)
  {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numRegions && activeRegions[tid] == 1) {
      size_t interval_index = scannedArray[tid];

      for (int i = 0; i < NDIM; ++i) {
        newActiveRegions[i * newNumRegions + interval_index] =
          dRegions[i * numRegions + tid];
        newActiveRegionsLength[i * newNumRegions + interval_index] =
          dRegionsLength[i * numRegions + tid];
      }

      dRegionsParentIntegral[interval_index] =
        dRegionsIntegral[tid /*+ numRegions*/];
      dRegionsParentError[interval_index] = dRegionsError[tid /*+ numRegions*/];

      // dRegionsParentIntegral[interval_index + newNumRegions] =
      // dRegionsIntegral[tid /*+ numRegions*/];
      // dRegionsParentError[interval_index + newNumRegions] = dRegionsError[tid
      // + numRegions];

      for (int i = 0; i < numOfDivisionOnDimension; ++i) {
        newActiveRegionsBisectDim[i * newNumRegions + interval_index] =
          subDividingDimension[tid];
      }
    }
  }

  template <typename T, int NDIM>
  __global__ void
  divideIntervalsGPU(T* genRegions,
                     T* genRegionsLength,
                     T* activeRegions,
                     T* activeRegionsLength,
                     int* activeRegionsBisectDim,
                     size_t numActiveRegions,
                     int numOfDivisionOnDimension)
  {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numActiveRegions) {

      int bisectdim = activeRegionsBisectDim[tid];
      size_t data_size = numActiveRegions * numOfDivisionOnDimension;

      for (int i = 0; i < numOfDivisionOnDimension; ++i) {
        for (int dim = 0; dim < NDIM; ++dim) {
          genRegions[i * numActiveRegions + dim * data_size + tid] =
            activeRegions[dim * numActiveRegions + tid];
          genRegionsLength[i * numActiveRegions + dim * data_size + tid] =
            activeRegionsLength[dim * numActiveRegions + tid];
        }
      }

      for (int i = 0; i < numOfDivisionOnDimension; ++i) {

        T interval_length =
          activeRegionsLength[bisectdim * numActiveRegions + tid] /
          numOfDivisionOnDimension;

        genRegions[bisectdim * data_size + i * numActiveRegions + tid] =
          activeRegions[bisectdim * numActiveRegions + tid] +
          i * interval_length;
        genRegionsLength[i * numActiveRegions + bisectdim * data_size + tid] =
          interval_length;
      }
    }
  }

  bool
  cudaMemoryTest()
  {
    // TODO: Doesn't this leak both h_a and d_a on every call?
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int* h_a = (int*)malloc(bytes);
    int* d_a;
    cudaMalloc((int**)&d_a, bytes);

    memset(h_a, 0, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    return true;
  }

  template <typename T, int NDIM>
  class Kernel {
    // these are also members of RegionList
    T* dRegionsError;
    T* dRegionsIntegral;
    T* dRegions;
    T* dRegionsLength;

    T* curr_hRegions;
    T* curr_hRegionsLength;

    T** hRegions;
    T** hRegionsLength;

    T* dParentsError;
    T* dParentsIntegral;
    T** hParentsError; // used only for batching phase 1 execution
    T** hParentsIntegral;

    //-----------------------------------
    T* polishedRegions;
    T* polishedRegionsLength;
    int* polishedRegionsSubDivDimension;
    T* polishedRegionsIntegral;
    T* polishedRegionsError;
    int* polishedRegionsDiv;

    //-----------------------------------

    T* highs;
    T* lows;

    Region<NDIM>* gRegionPool;
    PartitionManager<NDIM> partitionManager;
    int depthBeingProcessed;

    std::stringstream
      phase1out; // phase1 data on each region processed in an iteration
    std::stringstream finishedOutfile;

    std::stringstream out1;
    std::stringstream out2;
    std::stringstream out3;
    std::stringstream out4;
    std::stringstream out5;

    bool phase2;
    bool estimateHasConverged;
    int Final; // delete, not being used anymore
    int phase_I_type;
    int heuristicID;
    int fail; // 0 for satisfying ratio, 1 for not satisfying ratio, 2 for
              // running out of bad regions
    int phase2_failedblocks;
    T lastErr;
    T lastAvg;
    T secondTolastAvg;

    double errorest_change; // delete, not doing anything useful anymore
    double estimate_change;

    T* dlastErr;
    T* dlastAvg;
    int KEY, VERBOSE, outLevel;
    // delete numPolishedRegions, numFunctionEvaluations
    size_t numRegions, numPolishedRegions, h_numRegions, numFunctionEvaluations,
      numInActiveRegions;
    size_t fEvalPerRegion;
    int first_phase_maxregions;
    int max_globalpool_size;
    size_t host_current_list_id;
    bool mustFinish;
    const size_t numHostPartitions = 4;
    size_t partionSize[4]; // delete , not storing on cpu anymore
    double partionContributionsIntegral[4]; // delete, not storing on cpu
                                            // anymore
    double partionContributionsError[4]; // delete, not storing on cpu anymore

    size_t nextAvailRegionID; // make it a local variable to kernel fucntion, no
                              // need to bloat the class
    size_t nextAvailParentID; // same goes here
    size_t* parentIDs;        // same goes here

    bool phase2Ready = false;
    HostMemory<T> Host;
    DeviceMemory<T> Device;
    Rule<T> rule;
    Structures<T> constMem;
    int NUM_DEVICES;
    // Debug Msg
    char msg[256];

    std::ostream& log;
    double* generators = nullptr;

  public:
    T weightsum, avgsum, guess, chisq, chisum, chisqsum;

    void
    GetPtrsToArrays(double*& regions,
                    double*& regionsLength,
                    double*& regionsIntegral,
                    double*& regionsError,
                    double*& gener)
    {
      regions = dRegions;
      regionsLength = dRegionsLength;
      regionsIntegral = dRegionsIntegral;
      regionsError = dRegionsError;
      gener = generators;
    }

    void
    GetVars(size_t& numFuncEvals,
            size_t& numRegs,
            Structures<double>*& constMemory,
            int& nsets,
            int& depth)
    {
      numFuncEvals = fEvalPerRegion;
      numRegs = numRegions;
      constMemory = &constMem;
      nsets = rule.GET_NSETS();
      depth = depthBeingProcessed;
    }

    void
    GetVolumeBounds(double* lowBounds, double* highBounds)
    {
      lowBounds = lows;
      highBounds = highs;
    }

    int
    GetPhase2_failedblocks()
    {
      return phase2_failedblocks;
    }

    int
    GetPhase_I_type()
    {
      return phase_I_type;
    }

    double
    GetIntegral()
    {
      return lastAvg;
    }

    double
    GetError()
    {
      return lastErr;
    }

    int
    GetErrorFlag()
    {
      return fail;
    }

    double
    GetRatio(double epsrel, double epsabs)
    {
      return lastErr / MaxErr(lastAvg, epsrel, epsabs);
    }

    void
    SetPhase_I_type(int type)
    {
      phase_I_type = type;
    }

    void
    SetHeuristicID(int id)
    {
      heuristicID = id;
    }

    void
    SetVerbosity(const int verb)
    {
      outLevel = verb;
    }

    void
    SetFinal(const int _Final)
    {
      Final = _Final;
    }

    void
    ExpandcuArray(T*& array, int currentSize, int newSize)
    {
      T* temp = 0;
      int copy_size = std::min(currentSize, newSize);
      // printf("current size:%i, newSize:%i\n", currentSize, newSize);
      QuadDebug(Device.AllocateMemory((void**)&temp, sizeof(T) * newSize));
      CudaCheckError();
     /* QuadDebug(cudaMemcpy(
        temp, array, sizeof(T) * copy_size, cudaMemcpyDeviceToDevice));*/
      CudaCheckError();
      QuadDebug(Device.ReleaseMemory(array));
      array = temp;
    }

    size_t
    PredictSize(const int kernel_width,
                const int kernel_max_height,
                const size_t free_physmem,
                const size_t total_physmem)
    {

      size_t maxDeviceHeap = sizeof(T) * kernel_width * 2 * kernel_max_height;
      size_t regionsSize =
        sizeof(Region<NDIM>) * kernel_width * 2 * kernel_max_height;
      size_t reductionArrays =
        kernel_width * 2 * (sizeof(int) * 3 + sizeof(T) * 4);
      size_t setupSize = total_physmem - free_physmem;
      return maxDeviceHeap + regionsSize + reductionArrays + setupSize;
    }

    void
    ConfigureMemoryUtilization()
    {

      size_t free_physmem, total_physmem;
      QuadDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));

      first_phase_maxregions = FIRST_PHASE_MAXREGIONS;
      max_globalpool_size = MAX_GLOBALPOOL_SIZE;

      /*while(PredictSize(first_phase_maxregions, max_globalpool_size +
    SM_REGION_POOL_SIZE, free_physmem, total_physmem) < total_physmem){
    //printf("max_globalpool_size:%i can be increased\n",
    max_globalpool_size); max_globalpool_size += SM_REGION_POOL_SIZE;
    }*/

      /*while (PredictSize(2 * first_phase_maxregions,
                         max_globalpool_size,
                         free_physmem,
                         total_physmem) < total_physmem) {
        printf("first_phase_maxregions:%i can be increased\n",
               first_phase_maxregions);
        first_phase_maxregions += first_phase_maxregions;
      }*/

      /*
    std::cout<<max_globalpool_size<<","<<first_phase_maxregions<<std::endl;
    std::cout<<"first_phase_maxregions:"<<first_phase_maxregions<<std::endl;
    printf("Suggesting numBlocks:%i and max_global_size:%i vs numBlocks:%i and
    max_global_size:%i\n", first_phase_maxregions, max_globalpool_size,
    FIRST_PHASE_MAXREGIONS, MAX_GLOBALPOOL_SIZE);
      */
    }

    void
    SetPhase2(bool flag)
    {
      phase2 = flag;
    }

    Kernel(std::ostream& logerr = std::cout) : log(logerr)
    {
      mustFinish = false;
      numPolishedRegions = 0;
      dParentsError = nullptr;
      dParentsIntegral = nullptr;
      gRegionPool = nullptr;
      host_current_list_id = 0;
      errorest_change = 0.;
      estimate_change = 0.;
      /*double*/ // queued_reg_estimate = 0.;
      /*double*/ // queued_reg_errorest = 0.;
      estimateHasConverged = false;
      // arbitrary chose four iterations, if we want to create the biggest
      // dRegions we possibly can, four times
      hRegions = new double*[numHostPartitions];
      hRegionsLength = new double*[numHostPartitions];
      hParentsError = new double*[numHostPartitions]; // used only for batching
                                                      // phase 1 execution
      hParentsIntegral = new double*[numHostPartitions];

      ConfigureMemoryUtilization();

      polishedRegions = nullptr;
      polishedRegionsLength = nullptr;
      polishedRegionsSubDivDimension = nullptr;
      polishedRegionsIntegral = nullptr;
      polishedRegionsError = nullptr;
      polishedRegionsDiv = nullptr;

      /*QuadDebug(Device.AllocateMemory((void**)&gRegionPool,
                                      sizeof(Region<NDIM>) *
                                        first_phase_maxregions * 2 *
                                        max_globalpool_size));*/
      phase2 = false;
      phase2_failedblocks = 0;
      lastErr = 0;
      lastAvg = 0;
      Final = 0;
      fail = 1;
      weightsum = 0, avgsum = 0, guess = 0, chisq = 0, chisum = 0,
      chisqsum = 0; // only used when FINAL = 0 in Rcuhre
      numRegions = 0;
      numFunctionEvaluations = 0;
      // NDIM = 0;
      KEY = 0;
      phase_I_type =
        0; // breadth-first sub-region generation with good region fitler
      h_numRegions = 0;

      for (int i = 0; i < 4; i++) {
        partionSize[i] = 0;
        partionContributionsError[i] = 0.;
        partionContributionsIntegral[i] = 0.;
      }
    }

    ~Kernel()
    {
      CudaCheckError();
      // dRegions and dRegionsLength need to be freed after phase 1, since all
      // the info is stored in host memory

      QuadDebug(cudaFree(constMem._gpuG));
      QuadDebug(cudaFree(constMem._cRuleWt));
      QuadDebug(cudaFree(constMem._GPUScale));
      QuadDebug(cudaFree(constMem._GPUNorm));
      QuadDebug(cudaFree(constMem._gpuGenPos));
      QuadDebug(cudaFree(constMem._gpuGenPermGIndex));
      QuadDebug(cudaFree(constMem._gpuGenPermVarStart));
      QuadDebug(cudaFree(constMem._gpuGenPermVarCount));
      QuadDebug(cudaFree(constMem._cGeneratorCount));
      CudaCheckError();
    }

    void
    StringstreamToFile(std::string per_iteration,
                       std::string per_region,
                       int verbosity)
    {
      switch (verbosity) {
        case 1:
          PrintToFile(per_iteration,
                      "h" + std::to_string(heuristicID) + "_Per_iteration.csv");
          break;
        case 2:
          printf("Printing for verbosity 2\n");
          PrintToFile(per_iteration,
                      "h" + std::to_string(heuristicID) + "_Per_iteration.csv");
          PrintToFile(per_region, "Phase_1_regions.csv");
          break;
        default:
          break;
      }
    }

    size_t
    getNumActiveRegions()
    {
      return numRegions;
    }

    void
    setRegionsData(T* data, size_t size)
    {
      curr_hRegions = &data[0];
      curr_hRegionsLength = &data[size * NDIM];
      numRegions = size;
    }

    T*
    getRegions(size_t size, int startIndex)
    {
      T* newcurr_hRegionsAndLength = 0;
      newcurr_hRegionsAndLength = (T*)Host.AllocateMemory(
        &newcurr_hRegionsAndLength, 2 * sizeof(T) * size * NDIM);
      T *newcurr_hRegions = &newcurr_hRegionsAndLength[0],
        *newcurr_hRegionsLength = &newcurr_hRegionsAndLength[size * NDIM];
      // NOTE:Copy order is important
      for (int dim = 0; dim < NDIM; ++dim) {
        QuadDebug(cudaMemcpy(newcurr_hRegions + dim * size,
                             dRegions + dim * numRegions + startIndex,
                             sizeof(T) * size,
                             cudaMemcpyDeviceToHost));
        QuadDebug(cudaMemcpy(newcurr_hRegionsLength + dim * size,
                             dRegionsLength + dim * numRegions + startIndex,
                             sizeof(T) * size,
                             cudaMemcpyDeviceToHost));
      }
      return newcurr_hRegionsAndLength;
    }

    void
    InitKernel(int key, int verbose, int numDevices = 1)
    {
      // QuadDebug(cudaDeviceReset());
      // NDIM = dim;
      parentIDs = nullptr;
      numInActiveRegions = 0;
      nextAvailRegionID = 0;
      nextAvailParentID = 0;
      curr_hRegions = nullptr;
      curr_hRegionsLength = nullptr;
      depthBeingProcessed = 0;
      KEY = key;
      VERBOSE = verbose;
      NUM_DEVICES = numDevices;
      fEvalPerRegion = (1 + 2 * NDIM + 2 * NDIM + 2 * NDIM + 2 * NDIM +
                        2 * NDIM * (NDIM - 1) + 4 * NDIM * (NDIM - 1) +
                        4 * NDIM * (NDIM - 1) * (NDIM - 2) / 3 + (1 << NDIM));
      QuadDebug(cudaMemcpyToSymbol(dFEvalPerRegion,
                                   &fEvalPerRegion,
                                   sizeof(size_t),
                                   0,
                                   cudaMemcpyHostToDevice));
      rule.Init(NDIM, fEvalPerRegion, KEY, VERBOSE, &constMem);
      partitionManager.Init(&Host, &Device);
      // QuadDebug(Device.SetHeapSize());
    }

    void
    PrintOutfileHeaders()
    {
      if (outLevel >= 1) {
        out1 << "result, error, nregions" << std::endl;
      }
      if (outLevel >= 4) {
        out4 << "badregions, regions" << std::endl;
      }
    }

    void
    Phase_I_PrintFile(T epsrel, T epsabs)
    {

      if (outLevel >= 1 && phase_I_type == 0) {
        printf("OutLevel 1\n");
        PrintToFile(out1.str(), "Level_1.csv");
      }

      if (outLevel >= 3 && phase_I_type == 0) {
        printf("OutLevel 3\n");

        auto callback = [](T integral, T error, T rel) {
          return fabs(error / (rel * integral));
        };

        using func_pointer = double (*)(T integral, T error, T rel);
        func_pointer lambda_fp = callback;

        printf("About to display start_ratio\n");
        display(dRegionsIntegral + numRegions,
                dRegionsError + numRegions,
                epsrel,
                numRegions,
                lambda_fp,
                "start_ratio.csv",
                "result, error, initratio");

      } else if (outLevel >= 3) {
        out3 << "result, error, initratio" << std::endl;
        Region<NDIM>* tmp =
          (Region<NDIM>*)malloc(sizeof(Region<NDIM>) * numRegions);
        cudaMemcpy(tmp,
                   gRegionPool,
                   sizeof(Region<NDIM>) * numRegions,
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < numRegions; ++i) {
          double val = tmp[i].result.avg;
          double err = tmp[i].result.err;
          out3 << val << "," << err << "," << err / MaxErr(val, epsrel, epsabs)
               << std::endl;
        }
        PrintToFile(out3.str(), "start_ratio.csv");
        free(tmp);
      }
    }

    void
    PrintIteration(int* activeRegions,
                   int iteration,
                   size_t iter_nregions,
                   double leaves_estimate,
                   double leaves_errorest,
                   double iter_estimate,
                   double iter_errorest,
                   double iter_finished_estimate,
                   double iter_finished_errorest,
                   double queued_estimate,
                   double queued_errorest,
                   size_t unevaluated_nregions)
    {

      int* scannedArray = 0;
      QuadDebug(Device.AllocateMemory((void**)&scannedArray,
                                      sizeof(int) * iter_nregions));

      thrust::device_ptr<int> d_ptr =
        thrust::device_pointer_cast(activeRegions);
      thrust::device_ptr<int> scan_ptr =
        thrust::device_pointer_cast(scannedArray);
      thrust::exclusive_scan(d_ptr, d_ptr + iter_nregions, scan_ptr);

      int last_element;
      int dnumActiveRegions = 0;
      size_t dnumInActiveRegions = 0;

      QuadDebug(cudaMemcpy(&last_element,
                           activeRegions + iter_nregions - 1,
                           sizeof(int),
                           cudaMemcpyDeviceToHost));
      QuadDebug(cudaMemcpy(&dnumActiveRegions,
                           scannedArray + iter_nregions - 1,
                           sizeof(int),
                           cudaMemcpyDeviceToHost));

      if (last_element == 1)
        dnumActiveRegions++;
      dnumInActiveRegions = iter_nregions - dnumActiveRegions;

      if (iteration == 0)
        finishedOutfile
          << "HeuristicID, iteration, tot.est, tot.err, it.nreg, it.est, "
             "it.err,  it.fin.est, it.fin.err, it.fin.nreg, "
             "uneval.par.est, uneval.par.err, uneval.nreg\n";

      finishedOutfile << std::setprecision(17) << std::scientific << heuristicID
                      << "," << iteration << "," << leaves_estimate << ","
                      << leaves_errorest << "," << iter_nregions << ","
                      << iter_estimate << "," << iter_errorest << ","
                      << iter_finished_estimate << "," << iter_finished_errorest
                      << "," << dnumInActiveRegions << "," << queued_estimate
                      << "," << queued_errorest << "," << unevaluated_nregions
                      << "\n";

      std::cout << std::setprecision(17) << std::scientific << heuristicID
                << "," << iteration << "," << leaves_estimate << ","
                << leaves_errorest << "," << iter_nregions << ","
                << iter_estimate << "," << iter_errorest << ","
                << iter_finished_estimate << "," << iter_finished_errorest
                << "," << dnumInActiveRegions << "," << queued_estimate << ","
                << queued_errorest << "," << unevaluated_nregions << "\n";
      Device.ReleaseMemory(scannedArray);
    }

    void
    Phase_I_PrintFile(Volume<T, NDIM>* vol,
                      size_t iter_nregions,
                      int* activeRegions,
                      double leaves_estimate,
                      double leaves_errorest,
                      double iter_estimate,
                      double iter_errorest,
                      double iter_finished_estimate,
                      double iter_finished_errorest,
                      double queued_estimate,
                      double queued_errorest,
                      size_t unevaluated_nregions,
                      double epsrel,
                      double epsabs,
                      int iteration = 0)
    {

      if (outLevel < 1)
        return;

      // if(iteration != 28)
      //  return;

      PrintIteration(activeRegions,
                     iteration,
                     iter_nregions,
                     leaves_estimate,
                     leaves_errorest,
                     iter_estimate,
                     iter_errorest,
                     iter_finished_estimate,
                     iter_finished_errorest,
                     queued_estimate,
                     queued_errorest,
                     unevaluated_nregions);

      if (outLevel < 2)
        return;

      size_t numActiveRegions = 0;
      int* h_activeRegions = nullptr;
      h_activeRegions = (int*)Host.AllocateMemory(
        h_activeRegions,
        sizeof(int) *
          iter_nregions); //(int*)malloc(sizeof(int) * iter_nregions);

      double* curr_hRegionsIntegral = nullptr;
      double* curr_hRegionsError = nullptr;
      double* curr_ParentsIntegral = nullptr;
      double* curr_ParentsError = nullptr;

      double* h_highs = nullptr;
      double* h_lows = nullptr;

      curr_hRegionsIntegral = (double*)Host.AllocateMemory(
        curr_hRegionsIntegral,
        sizeof(double) *
          iter_nregions); //(double*)malloc(sizeof(double) * iter_nregions);
      curr_hRegionsError = (double*)Host.AllocateMemory(
        curr_hRegionsError,
        sizeof(double) *
          iter_nregions); //(double*)malloc(sizeof(double) * iter_nregions);
      curr_ParentsIntegral = (double*)Host.AllocateMemory(
        curr_ParentsIntegral,
        sizeof(double) *
          ceil(iter_nregions /
               2)); //(double*)malloc(sizeof(double) * iter_nregions);
      curr_ParentsError = (double*)Host.AllocateMemory(
        curr_ParentsError,
        sizeof(double) *
          ceil(iter_nregions /
               2)); //(double*)malloc(sizeof(double) * iter_nregions);
      h_highs = (double*)Host.AllocateMemory(
        h_highs, sizeof(double) * NDIM); //(double*)malloc(sizeof(double)*NDIM);
      h_lows = (double*)Host.AllocateMemory(
        h_lows, sizeof(double) * NDIM); //(double*)malloc(sizeof(double)*NDIM);

      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_hRegionsIntegral,
                           dRegionsIntegral,
                           sizeof(double) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_hRegionsError,
                           dRegionsError,
                           sizeof(double) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();

      // printf("Displaying Estimates\n");
      // display<double, 2>(iter_nregions, dRegionsIntegral, dRegionsError);

      QuadDebug(cudaMemcpy(curr_ParentsIntegral,
                           dParentsIntegral,
                           sizeof(double) * ceil(iter_nregions / 2),
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_ParentsError,
                           dParentsError,
                           sizeof(double) * ceil(iter_nregions / 2),
                           cudaMemcpyDeviceToHost));
      CudaCheckError();

      // printf("iteration:%i Displaying Parents\n", iteration);
      // display<double, 2>(iter_nregions/2, dParentsIntegral, dParentsError);

      QuadDebug(cudaMemcpy(h_activeRegions,
                           activeRegions,
                           sizeof(int) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(
        h_highs, highs, sizeof(double) * NDIM, cudaMemcpyDeviceToHost));

      QuadDebug(cudaMemcpy(
        h_lows, lows, sizeof(double) * NDIM, cudaMemcpyDeviceToHost));

      CudaCheckError();

      curr_hRegions = (double*)Host.AllocateMemory(
        curr_hRegions,
        sizeof(double) * iter_nregions *
          NDIM); //(double*)malloc(sizeof(double) * iter_nregions * NDIM);
      curr_hRegionsLength = (double*)Host.AllocateMemory(
        curr_hRegionsLength,
        sizeof(double) * iter_nregions *
          NDIM); //(double*)malloc(sizeof(double) * iter_nregions * NDIM);
      // free_bounds_needed = true;
      QuadDebug(cudaMemcpy(curr_hRegions,
                           dRegions,
                           sizeof(double) * iter_nregions * NDIM,
                           cudaMemcpyDeviceToHost));
      QuadDebug(cudaMemcpy(curr_hRegionsLength,
                           dRegionsLength,
                           sizeof(double) * iter_nregions * NDIM,
                           cudaMemcpyDeviceToHost));

      CudaCheckError();

      if (iteration == 0) {
        phase1out << "iteration, id, parentID, estimate, errorest, parEst, "
                     "parErr, active,";
        for (size_t i = 0; i < NDIM; i++) {
          std::string dim = std::to_string(i);
          phase1out << "dim" + dim + "low, dim" + dim + "high";
          if (i == NDIM - 1)
            phase1out << "\n";
          else
            phase1out << ",";
        }
      }

      auto GetParentIndex = [](size_t selfIndex, size_t num_regions) {
        size_t inRightSide = (2 * selfIndex >= num_regions);
        // size_t inLeftSide =  (0 >= inRightSide);
        size_t parIndex = selfIndex - inRightSide * (num_regions * .5);
        return parIndex;
      };

      for (size_t regnIndex = 0; regnIndex < iter_nregions; regnIndex++) {
        size_t parentID = 0;
        size_t parentIndex = GetParentIndex(regnIndex, iter_nregions);

        if (iter_nregions > 1) {
          parentID = regnIndex < iter_nregions / 2 ?
                       parentIDs[regnIndex] :
                       parentIDs[regnIndex - iter_nregions / 2];
        }

        double ratio = curr_hRegionsError[regnIndex] /
                       (epsrel * abs(curr_hRegionsIntegral[regnIndex]));

        if (iter_nregions > 1)
          phase1out << std::setprecision(17) << std::scientific << iteration
                    << "," << nextAvailRegionID + regnIndex << "," << parentID;
        else
          phase1out << std::setprecision(17) << std::scientific << iteration
                    << "," << nextAvailRegionID + regnIndex << "," << -1;

        phase1out << "," << curr_hRegionsIntegral[regnIndex] << ","
                  << curr_hRegionsError[regnIndex] << ","
                  << curr_ParentsIntegral[parentIndex] << ","
                  << curr_ParentsError[parentIndex] << ","
                  << h_activeRegions[regnIndex] << ",";

        // if (h_activeRegions[regnIndex] == 1)
        numActiveRegions++;

        for (size_t dim = 0; dim < NDIM; dim++) {
          double low =
            ScaleValue(curr_hRegions[dim * iter_nregions + regnIndex],
                       h_lows[dim],
                       h_highs[dim]);

          double high =
            ScaleValue(curr_hRegions[dim * iter_nregions + regnIndex] +
                         curr_hRegionsLength[dim * iter_nregions + regnIndex],
                       h_lows[dim],
                       h_highs[dim]);

          if (dim == NDIM - 1) {
            phase1out << std::setprecision(17) << std::scientific << low << ","
                      << high << "\n";
          } else {
            phase1out << std::setprecision(17) << std::scientific << low << ","
                      << high << ",";
          }
        }

        // phase1out << "\n";
      }

      CudaCheckError();
      if (iteration != 0) {
        Host.ReleaseMemory(parentIDs);
      }

      parentIDs = (size_t*)Host.AllocateMemory(
        parentIDs, sizeof(size_t) * numActiveRegions);
      CudaCheckError();

      size_t nextActiveRegion = 0;
      for (size_t i = 0; i < iter_nregions; i++) {
        parentIDs[nextActiveRegion] = nextAvailRegionID + i;
        nextActiveRegion++;
      }
      nextAvailRegionID += iter_nregions;

      Host.ReleaseMemory(curr_hRegionsError);
      Host.ReleaseMemory(curr_hRegionsIntegral);
      Host.ReleaseMemory(curr_hRegions);
      Host.ReleaseMemory(curr_hRegionsLength);
      Host.ReleaseMemory(curr_ParentsIntegral);
      Host.ReleaseMemory(curr_ParentsError);
      Host.ReleaseMemory(h_activeRegions);
      Host.ReleaseMemory(h_highs);
      Host.ReleaseMemory(h_lows);
    }

    // void Phase_IΙ_Print_File(double integral, double error, double epsrel,
    // double epsabs, int regionCnt, int* dRegionsNumRegion, size_t size){
    void
    Phase_II_PrintFile(T integral,
                       T error,
                       T epsrel,
                       T epsabs,
                       int regionCnt,
                       int* dRegionsNumRegion,
                       Region<NDIM>* hgRegionsPhase1,
                       size_t size)
    {

      if (outLevel >= 1) {
        out1 << integral << "," << error << "," << (regionCnt - size)
             << std::endl;
        PrintToFile(out1.str(), "Level_1.csv");
      }

      if (outLevel >= 2) {
        auto callback = [](T integral, T error, T rel) {
          return fabs(error / (rel * integral));
        };

        using func_pointer = double (*)(T integral, T error, T rel);
        func_pointer lambda_fp = callback;

        display(dRegionsIntegral,
                dRegionsError,
                epsrel,
                size,
                lambda_fp,
                "end_ratio.csv",
                "result, error, end_ratio");
        display(dRegionsNumRegion, size, "numRegions.csv", "nregions");
      }

      if (outLevel >= 4) {
        // this outLevel will only work for phasetype = 1
        Region<NDIM>* cgRegionPool = 0;
        int* RegionsNumRegion = 0;
        cgRegionPool = (Region<NDIM>*)malloc(sizeof(Region<NDIM>) * size *
                                             max_globalpool_size);
        RegionsNumRegion = (int*)malloc(sizeof(int) * size);

        QuadDebug(cudaMemcpy(cgRegionPool,
                             gRegionPool,
                             sizeof(Region<NDIM>) * size * max_globalpool_size,
                             cudaMemcpyDeviceToHost));

        // printf("Inside print file for phase 2 cgRegionPool[4096]\n");

        CudaCheckError();
        QuadDebug(cudaMemcpy(RegionsNumRegion,
                             dRegionsNumRegion,
                             sizeof(int) * size,
                             cudaMemcpyDeviceToHost));

        if (phase_I_type == 0) {
          OutputPhase2Regions(cgRegionPool,
                              curr_hRegions,
                              curr_hRegionsLength,
                              RegionsNumRegion,
                              size,
                              size * max_globalpool_size);
        } else {
          OutputPhase2Regions(
            cgRegionPool, hgRegionsPhase1, RegionsNumRegion, size, 0);
        }
      }
    }

    template <class K, size_t numArrays>
    void
    display(size_t arraySize, ...)
    {
      va_list params;
      va_start(params, arraySize);

      std::array<K*, numArrays> hArrays;

      for (auto& array : hArrays) {
        array = (K*)malloc(sizeof(K) * arraySize);
        cudaMemcpy(array,
                   va_arg(params, K*),
                   sizeof(K) * arraySize,
                   cudaMemcpyDeviceToHost);
      }

      va_end(params);

      auto PrintSameIndex = [hArrays](size_t index) {
        std::cout << index << "): ";
        for (auto& array : hArrays) {
          std::cout << array[index] << "\t";
        }
        std::cout << std::endl;
      };

      for (size_t i = 0; i < arraySize; ++i)
        PrintSameIndex(i);

      for (auto& array : hArrays)
        free(array);
    }

    size_t
    GetGPUMemNeededForNextIteration_CallBeforeSplit()
    {
      // if we split everything in 2 (means if we call GenerateActiveIntervals)
      // will we have enough memory to compute at least the leaves for the next
      // iteration?

      /*size_t nextIter_dParentsIntegral_size = 2*numRegions ; //parents are
      duplicated size_t nextIter_dParentsError_size = 2*numRegions; //parents
      are duplicated size_t nextIter_dRegions_size = 2*numRegions * NDIM; size_t
      nextIter_dRegionsLength_size = 2*numRegions * NDIM; size_t
      nextIter_dRegionsIntegral_size = 2*numRegions; size_t
      nextIter_dRegionsError_size = 2*numRegions;

      //we also need to worry about the temporary arrays that are created to do
      the copies size_t temp_RegionsLength = numRegions*NDIM; size_t
      temp_Regions = numRegions*NDIM; size_t temp_dParentsIntegral_size =
      2*numRegions ; //parents are duplicated size_t temp_dParentsError_size =
      2*numRegions; //parents are duplicated

      size_t tempBisectDim = numRegions;

      size_t _activeRegions = 2*numRegions;
      size_t _subDividingDimension = 2*numRegions;

      size_t Ints_Size = 4*(_activeRegions + _subDividingDimension +
      tempBisectDim); size_t Doubles_Size = 8*(nextIter_dParentsIntegral_size +
          nextIter_dParentsError_size + nextIter_dRegions_size +
          nextIter_dRegionsLength_size +
          nextIter_dRegionsIntegral_size +
          nextIter_dRegionsError_size+
          temp_RegionsLength + temp_RegionsLength +
          temp_Regions +
          temp_dParentsIntegral_size +
          temp_dParentsError_size);
      //the above doesn't consider the intermediate memory needed
      size_t scatchSpaceInGenerateActiveIntervals = (Ints_Size +
      Doubles_Size)/2; return Ints_Size + Doubles_Size +
      scatchSpaceInGenerateActiveIntervals;*/

      // doubles needed to classify and divide now
      //----------------------------------------------------------
      size_t scanned = numRegions;
      size_t newActiveRegions = numRegions * NDIM;
      size_t newActiveRegionsLength = numRegions * NDIM;
      size_t parentExpansionEstimate = numRegions;
      size_t parentExpansionErrorest = numRegions;
      size_t genRegions = numRegions * NDIM * 2;
      size_t genRegionsLength = numRegions * NDIM * 2;

      // ints needed to classify and divide now
      size_t activeBisectDim = numRegions;
      //----------------------------------------------------------
      // doubles needed for sampling next iteration
      size_t regions = 2 * numRegions * NDIM;
      size_t regionsLength = 2 * numRegions * NDIM;
      size_t regionsIntegral = 2 * numRegions;
      size_t regionsError = 2 * numRegions;
      size_t parentsIntegral = numRegions;
      size_t parentsError = numRegions;
      // ints needed for sampling next iteration
      size_t subDivDim = 2 * numRegions;

      //----------------------------------------------------------

      // we also need to worry about the temporary arrays that are created to do
      // the copies

      size_t Ints_Size = 4 * (activeBisectDim + subDivDim + scanned);
      size_t Doubles_Size =
        8 * (newActiveRegions + newActiveRegionsLength +
             parentExpansionEstimate + parentExpansionErrorest + genRegions +
             genRegionsLength + regions + regionsLength + regionsIntegral +
             regionsError + parentsIntegral + parentsError);
      // the above doesn't consider the intermediate memory needed
      return Ints_Size + Doubles_Size;
    }

    size_t
    GetGPUMemNeededForNextIteration()
    {
      // numRegions implies the number of regions processed in next iteration
      // so it's supposed to be called only after GenerateIntervals has been
      // called
      int numOfDivisionOnDimension = 2;
      size_t nextIter_dParentsIntegral_size = numRegions /** 2*/;
      size_t nextIter_dParentsError_size = numRegions /** 2*/;
      size_t nextIter_dRegions_size = numRegions * NDIM;
      size_t nextIter_dRegionsLength_size = numRegions * NDIM;
      size_t nextIter_dRegionsIntegral_size = numRegions;
      size_t nextIter_dRegionsError_size = numRegions;
      size_t nextIter_newActiveRegions_size =
        numRegions * NDIM * numOfDivisionOnDimension;
      size_t nextIter_newActiveRegionsLength_size =
        numRegions * NDIM * numOfDivisionOnDimension;

      size_t nextIter_newActiveRegionsBisectDim_size = numRegions;
      size_t nextIter_activeRegions_size = numRegions;
      size_t nextIter_subdividingDimension_size = numRegions;
      size_t nextIter_scannedArray_size = numRegions;

      size_t Doubles_Size =
        sizeof(double) *
        (nextIter_dParentsIntegral_size + nextIter_dParentsError_size +
         nextIter_dRegionsError_size + nextIter_dRegionsIntegral_size +
         nextIter_dRegionsError_size + nextIter_dRegions_size +
         nextIter_dRegionsLength_size + nextIter_newActiveRegions_size +
         nextIter_newActiveRegionsLength_size);

      size_t Ints_Size =
        sizeof(int) *
        (nextIter_activeRegions_size + nextIter_subdividingDimension_size +
         nextIter_scannedArray_size + nextIter_newActiveRegionsBisectDim_size);

      return Ints_Size + Doubles_Size;
    }

    void
    GenerateInitialRegions()
    {
      curr_hRegions = (T*)Host.AllocateMemory(&curr_hRegions, sizeof(T) * NDIM);
      curr_hRegionsLength =
        (T*)Host.AllocateMemory(&curr_hRegionsLength, sizeof(T) * NDIM);

      for (int dim = 0; dim < NDIM; ++dim) {
        curr_hRegions[dim] = 0;
#if GENZ_TEST == 1
        curr_hRegionsLength[dim] = b[dim];
#else
        curr_hRegionsLength[dim] = 1;
#endif
      }

      QuadDebug(Device.AllocateMemory((void**)&dRegions, sizeof(T) * NDIM));
      QuadDebug(
        Device.AllocateMemory((void**)&dRegionsLength, sizeof(T) * NDIM));

      QuadDebug(cudaMemcpy(
        dRegions, curr_hRegions, sizeof(T) * NDIM, cudaMemcpyHostToDevice));
      QuadDebug(cudaMemcpy(dRegionsLength,
                           curr_hRegionsLength,
                           sizeof(T) * NDIM,
                           cudaMemcpyHostToDevice));

      size_t numThreads = 512;
      // this has been changed temporarily, do not remove
      size_t numOfDivisionPerRegionPerDimension = 4;
      if (NDIM == 5)
        numOfDivisionPerRegionPerDimension = 2;
      if (NDIM == 6)
        numOfDivisionPerRegionPerDimension = 2;
      if (NDIM == 7)
        numOfDivisionPerRegionPerDimension = 2;
      if (NDIM > 7)
        numOfDivisionPerRegionPerDimension = 2;
      if (NDIM > 10)
        numOfDivisionPerRegionPerDimension = 1;

      depthBeingProcessed = log2(numOfDivisionPerRegionPerDimension) * NDIM;
      // size_t numOfDivisionPerRegionPerDimension = 1;

      size_t numBlocks = (size_t)ceil(
        pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM) / numThreads);
      numRegions = (size_t)pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM);

      T* newRegions = 0;
      T* newRegionsLength = 0;
      QuadDebug(Device.AllocateMemory((void**)&newRegions,
                                      sizeof(T) * numRegions * NDIM));
      QuadDebug(Device.AllocateMemory((void**)&newRegionsLength,
                                      sizeof(T) * numRegions * NDIM));

      generateInitialRegions<T><<<numBlocks, numThreads, NDIM * sizeof(T)>>>(
        dRegions,
        dRegionsLength,
        1,
        newRegions,
        newRegionsLength,
        numRegions,
        numOfDivisionPerRegionPerDimension,
        NDIM);

      QuadDebug(Device.ReleaseMemory((void*)dRegions));
      QuadDebug(Device.ReleaseMemory((void*)dRegionsLength));

      dRegions = newRegions;
      dRegionsLength = newRegionsLength;
      QuadDebug(cudaMemcpy(dRegions,
                           newRegions,
                           sizeof(T) * numRegions * NDIM,
                           cudaMemcpyDeviceToDevice));
      QuadDebug(cudaMemcpy(dRegionsLength,
                           newRegionsLength,
                           sizeof(T) * numRegions * NDIM,
                           cudaMemcpyDeviceToDevice));
      Host.ReleaseMemory(curr_hRegions);
      Host.ReleaseMemory(curr_hRegionsLength);
      // free(curr_hRegionsLength);
      // curr_hRegions = nullptr;
      // curr_hRegionsLength = nullptr;
    }

    size_t
    GenerateActiveIntervals(int* activeRegions,
                            int* subDividingDimension,
                            T* dRegionsIntegral,
                            T* dRegionsError,
                            T*& dParentsIntegral,
                            T*& dParentsError)
    {
      nvtxRangePush("Compute numActive Regions");
      int* scannedArray = 0; // de-allocated at the end of this function
      QuadDebug(
        Device.AllocateMemory((void**)&scannedArray, sizeof(int) * numRegions));

      thrust::device_ptr<int> d_ptr =
        thrust::device_pointer_cast(activeRegions);

      CudaCheckError();

      thrust::device_ptr<int> scan_ptr =
        thrust::device_pointer_cast(scannedArray);
      nvtxRangePush("Exclusive scan");
      thrust::exclusive_scan(d_ptr, d_ptr + numRegions, scan_ptr);
       nvtxRangePop();
      CudaCheckError();

      int last_element;
      size_t numActiveRegions = 0;

      QuadDebug(cudaMemcpy(&last_element,
                           activeRegions + numRegions - 1,
                           sizeof(int),
                           cudaMemcpyDeviceToHost));
      QuadDebug(cudaMemcpy(&numActiveRegions,
                           scannedArray + numRegions - 1,
                           sizeof(int),
                           cudaMemcpyDeviceToHost));

      if (last_element == 1)
        numActiveRegions++;
    
      nvtxRangePop();
      
      numInActiveRegions = numRegions - numActiveRegions;
      
      // printf("Bad Reginos %lu/%lu\n", numActiveRegions, numRegions);
      if (outLevel >= 4)
        out4 << numActiveRegions << "," << numRegions << std::endl;

      if (numActiveRegions > 0) {
        nvtxRangePush("Aligning Regions");
        int numOfDivisionOnDimension = 2;

        int* newActiveRegionsBisectDim = 0;
        T *newActiveRegions = 0, *newActiveRegionsLength =
                                   0; // de-allocated at the end of the function

        cudaMalloc((void**)&newActiveRegions,
                   sizeof(T) * numActiveRegions * NDIM);
        cudaMalloc((void**)&newActiveRegionsLength,
                   sizeof(T) * numActiveRegions * NDIM);
        cudaMalloc((void**)&newActiveRegionsBisectDim,
                   sizeof(int) * numActiveRegions * numOfDivisionOnDimension);
        CudaCheckError();

     

        ExpandcuArray(dParentsIntegral, numRegions / 2, numActiveRegions);
        CudaCheckError();
        ExpandcuArray(dParentsError, numRegions / 2, numActiveRegions);
       
        CudaCheckError();
        size_t numThreads = BLOCK_SIZE;
        size_t numBlocks =
          numRegions / numThreads + ((numRegions % numThreads) ? 1 : 0);

        cudaDeviceSynchronize();
        // printf("Populating dParents with the results of dRegionsIntegral\n");
        //printf("Aligning %lu numRegions\n", numRegions);
        
        alignRegions<T, NDIM>
          <<<numBlocks, numThreads>>>(dRegions,
                                      dRegionsLength,
                                      activeRegions,
                                      dRegionsIntegral,
                                      dRegionsError,
                                      dParentsIntegral,
                                      dParentsError,
                                      subDividingDimension,
                                      scannedArray,
                                      newActiveRegions,
                                      newActiveRegionsLength,
                                      newActiveRegionsBisectDim,
                                      numRegions,
                                      numActiveRegions,
                                      numOfDivisionOnDimension);

        // cudaDeviceSynchronize();
        CudaCheckError();
        nvtxRangePop();
        T *genRegions = 0, *genRegionsLength = 0;
        numBlocks = numActiveRegions / numThreads +
                    ((numActiveRegions % numThreads) ? 1 : 0);
        
        QuadDebug(Device.ReleaseMemory(dRegions));
        QuadDebug(Device.ReleaseMemory(dRegionsLength));
        QuadDebug(Device.ReleaseMemory(scannedArray));

        // printf("need to allocate %lu bytes\n", sizeof(T) * numActiveRegions *
        // NDIM * numOfDivisionOnDimension *2); printf("Amount of free
        // memory:%lu\n", Device.GetAmountFreeMem());
        nvtxRangePush("dividing Intervals");
        QuadDebug(cudaMalloc((void**)&genRegions,
                             sizeof(T) * numActiveRegions * NDIM *
                               numOfDivisionOnDimension));
        QuadDebug(cudaMalloc((void**)&genRegionsLength,
                             sizeof(T) * numActiveRegions * NDIM *
                               numOfDivisionOnDimension));
        CudaCheckError();
        //printf("Dividing %lu regions\n", numActiveRegions);
        
        divideIntervalsGPU<T, NDIM>
          <<<numBlocks, numThreads>>>(genRegions,
                                      genRegionsLength,
                                      newActiveRegions,
                                      newActiveRegionsLength,
                                      newActiveRegionsBisectDim,
                                      numActiveRegions,
                                      numOfDivisionOnDimension);

        CudaCheckError();
        
        QuadDebug(Device.ReleaseMemory(newActiveRegions));
        QuadDebug(Device.ReleaseMemory(newActiveRegionsLength));
        QuadDebug(Device.ReleaseMemory(newActiveRegionsBisectDim));
        CudaCheckError();
        numRegions = numActiveRegions * numOfDivisionOnDimension;

        dRegions = genRegions;
        dRegionsLength = genRegionsLength;
        nvtxRangePop();
        cudaDeviceSynchronize();
        CudaCheckError();
      } else {
        numRegions = 0;
      }

      return numInActiveRegions;
    }

    void
    GetNextThreshold(double min,
                     double max,
                     int rightDirection,
                     double& current)
    {
      if (rightDirection) {
        double diff = abs(max - current);
        current += diff * .5f;
      } else {
        double diff = abs(min - current);
        current -= diff * .5f;
      }
    }

    bool
    AdjustErrorThreshold(double ErrorestMarkedFinished,
                         double MaxErrorestToAllowedToFinish,
                         double percentRegionsToKeep,
                         int& currentDirection,
                         double& lastThreshold,
                         double& threshold,
                         double& minThreshold,
                         double& maxThreshold,
                         int& numDirectionChanges)
    {
      // printf("adjusting threshold with min max:%15e, %.15e\n", minThreshold,
      // maxThreshold);
      int priorDirection = currentDirection;
      if (ErrorestMarkedFinished > MaxErrorestToAllowedToFinish ||
          percentRegionsToKeep < .5) {
        if (currentDirection == 1) {
          numDirectionChanges++;
          if (lastThreshold < threshold) {
            minThreshold = lastThreshold;
            maxThreshold = threshold;
          }
        } else
          lastThreshold = threshold;

        currentDirection = 0;
        GetNextThreshold(
          minThreshold, maxThreshold, currentDirection, threshold);
      } else if (percentRegionsToKeep > .5) {
        if (currentDirection == 0) {
          numDirectionChanges++;
          if (threshold < lastThreshold) {
            maxThreshold = lastThreshold;
            minThreshold = threshold;
          }

        } else
          lastThreshold = threshold;

        currentDirection = 1;
        GetNextThreshold(minThreshold,
                         maxThreshold,
                         currentDirection,
                         threshold); // the flag was priorly set to zero
      }
      // printf("new from within adjustment min max:%15e, %.15e\n",
      // minThreshold, maxThreshold);
      return currentDirection != priorDirection &&
             priorDirection != -1; // return whether there is a direction change
    }

    size_t
    ComputeNumUnPolishedRegions(thrust::device_ptr<int> d_ptr,
                                int* unpolishedRegions,
                                thrust::device_ptr<int> scan_ptr,
                                int* scannedArray,
                                size_t numRegions)
    {
      size_t _numUnPolishedRegions = 0;
      _numUnPolishedRegions = thrust::reduce(d_ptr, d_ptr + numRegions);
      return _numUnPolishedRegions;
    }

    void
    HSClassify(T* dRegionsIntegral,
               T* dRegionsError,
               int* subDividingDimension,
               T*& dParentsIntegral,
               T*& dParentsError,
               int*& activeRegions,
               double& integral,
               double& error,
               size_t& nregions,
               double iterEstimate,
               double iterErrorest,
               double iter_finished_estimate,
               double iter_finished_errorest,
               double leaves_estimate,
               double leaves_errorest,
               double epsrel,
               double epsabs,
               int iteration)
    {

      int requiredDigits =
        ceil(log10(1 / epsrel)); // this is no longer used in RelErrClassify,
                                 // move below when used by Filter
      estimateHasConverged =
        estimateHasConverged == false ?
          (iteration >= 2 ?
             sigDigitsSame(
               lastAvg, secondTolastAvg, leaves_estimate, requiredDigits) :
             false) :
          true;
      // printf("it:%i estimateHasConverged:%i\n", iteration,
      // estimateHasConverged);
      secondTolastAvg = lastAvg;
      lastAvg = leaves_estimate;
      lastErr = leaves_errorest;
       /*printf("it:%i need %lu have %lu which is %f percent numRegions:%lu\n",
       iteration, GetGPUMemNeededForNextIteration_CallBeforeSplit(),
       Device.GetAmountFreeMem(),
       (double)GetGPUMemNeededForNextIteration_CallBeforeSplit()/(double)Device.GetAmountFreeMem(),
       numRegions);*/
      if (phase2 == true || (GetGPUMemNeededForNextIteration_CallBeforeSplit() <
                               Device.GetAmountFreeMem() &&
                             !estimateHasConverged))
        return;
      // printf("Will attempt Filtering\n");
      double targetError = abs(leaves_estimate) * epsrel;
      size_t numThreads = BLOCK_SIZE;

      size_t numBlocks =
        numRegions / numThreads + ((numRegions % numThreads) ? 1 : 0);

      double MaxPercentOfErrorBudget = .25;
      double acceptableThreshold = 0.;
      int* unpolishedRegions = 0;
      int* scannedArray = 0; // de-allocated at the end of this function
      size_t targetRegionNum = numRegions / 2;

      double ErrThreshold =
        iterErrorest / (numRegions); // starts out as avg value
       /*printf("initial error threshold:%a = %.15e iterErrorest:%.15e\n",
       ErrThreshold, ErrThreshold, iterErrorest);*/
      double lastThreshold = ErrThreshold;
      size_t numActiveRegions = numRegions;
      double iter_polished_errorest = 0.;
      double iter_polished_estimate = 0.;
      int numDirectionChanges = 0;

      QuadDebug(Device.AllocateMemory((void**)&unpolishedRegions,
                                      sizeof(int) * numRegions));

      thrust::device_ptr<int> d_ptr =
        thrust::device_pointer_cast(unpolishedRegions);
      int direction = -1;

      thrust::device_ptr<double> d_ptrE =
        thrust::device_pointer_cast(dRegionsError);

      thrust::pair<thrust::device_vector<double>::iterator,
                   thrust::device_vector<double>::iterator>
        tuple;
      auto __tuple = thrust::minmax_element(d_ptrE, d_ptrE + numRegions);
      double min = *__tuple.first;
      double max = *__tuple.second;
      thrust::device_ptr<double> wrapped_ptr;
      thrust::device_ptr<int> wrapped_mask;
      bool directionChange = false;
      int maxDirectionChanges = 9;

      do {
        iter_polished_estimate = 0.;
        iter_polished_errorest = 0.;

        if (numDirectionChanges >= maxDirectionChanges) {
          ErrThreshold =
            acceptableThreshold; // saved as last resort, not original target
                                 // for reamining errorest budget percentage to
                                 // cover but good enough
        } else if (numDirectionChanges > 2 && numDirectionChanges <= 9 &&
                   directionChange) {
          MaxPercentOfErrorBudget =
            numDirectionChanges > 1 ?
              MaxPercentOfErrorBudget + .1 :
              MaxPercentOfErrorBudget; // if we are doing a lot of back and
                                       // forth, we must relax the requirements
        }

        numActiveRegions = 0;
        Filter<<<numBlocks, numThreads>>>(dRegionsError,
                                          unpolishedRegions,
                                          activeRegions,
                                          numRegions,
                                          ErrThreshold);
        cudaDeviceSynchronize();
        CudaCheckError();

        numActiveRegions = ComputeNumUnPolishedRegions(
          d_ptr, unpolishedRegions, nullptr, scannedArray, numRegions);
        // printf("number active regions:%lu with potential threshold %.15e\n",
        // numActiveRegions, ErrThreshold);
        if (numActiveRegions <= targetRegionNum) {

          wrapped_mask = thrust::device_pointer_cast(unpolishedRegions);
          wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral);
          iter_polished_estimate =
            iterEstimate - iter_finished_estimate -
            thrust::inner_product(thrust::device,
                                  wrapped_ptr,
                                  wrapped_ptr + numRegions,
                                  wrapped_mask,
                                  0.);

          wrapped_ptr = thrust::device_pointer_cast(dRegionsError);
          iter_polished_errorest =
            iterErrorest - iter_finished_errorest -
            thrust::inner_product(thrust::device,
                                  wrapped_ptr,
                                  wrapped_ptr + numRegions,
                                  wrapped_mask,
                                  0.);

          if ((iter_polished_errorest <=
                 MaxPercentOfErrorBudget * (targetError - error) ||
               numDirectionChanges == maxDirectionChanges)) {
            integral += iter_polished_estimate;
            error += iter_polished_errorest;
            //printf("Found it %.15f +- %.15f\n", integral, error);
            break;
          } else if (iter_polished_errorest <= .95 * (targetError - error)) {
            acceptableThreshold = ErrThreshold;
            // printf("Found acceptableThreshold:%.15e\n", acceptableThreshold);
          }
        }
        double unpolishedPercentage =
          (double)(numActiveRegions) / (double)numRegions;
        // change that, intent is not clear should be returning directionChange
        // and have the name AdjustErrorThreshold
        directionChange =
          AdjustErrorThreshold(iter_polished_errorest,
                               MaxPercentOfErrorBudget * (targetError - error),
                               unpolishedPercentage,
                               direction,
                               lastThreshold,
                               ErrThreshold,
                               min,
                               max,
                               numDirectionChanges);

        if (numDirectionChanges == maxDirectionChanges &&
            acceptableThreshold == 0.) {
          // printf("Could not do it numDirectionChanges:%i
          // acceptableThreshold:%.15e estimateHasConverged:%i\n",
          // numDirectionChanges, acceptableThreshold, estimateHasConverged);
          QuadDebug(Device.ReleaseMemory(unpolishedRegions));
          if (!estimateHasConverged ||
              GetGPUMemNeededForNextIteration_CallBeforeSplit() >=
                Device.GetAmountFreeMem())
            mustFinish = true;
          return;
        }

      } while (numActiveRegions > targetRegionNum ||
               iter_polished_errorest >
                 MaxPercentOfErrorBudget * (targetError - error) ||
               error > targetError);

      if (numActiveRegions == numRegions) {
        // printf("Didn't filter out anything\n");
        mustFinish = true;
        // RevertFinishedStatus<<<numBlocks, numThreads>>>(activeRegions,
        // numRegions);
        // cudaDeviceSynchronize();
        QuadDebug(Device.ReleaseMemory(unpolishedRegions));
      } else {
        QuadDebug(Device.ReleaseMemory(activeRegions));
        activeRegions = unpolishedRegions;
        CudaCheckError();
      }
    }

    void
    AllocVolArrays(Volume<T, NDIM>* vol)
    {
      /*
        this is invoked by IntegateFirstPhase and doesn't need to called by user
        is currently present in the test RegionSampling due to ~Kernel's
        destruction of highs, lows if AllocVolArrays is never called, cudaFree
        will cause error
      */
      //nvtxRangePush("AllocVolArrays");
      cudaMalloc((void**)&lows, sizeof(T) * NDIM);
      cudaMalloc((void**)&highs, sizeof(T) * NDIM);

      if (vol) {
        cudaMemcpy(lows, vol->lows, sizeof(T) * NDIM, cudaMemcpyHostToDevice);
        cudaMemcpy(highs, vol->highs, sizeof(T) * NDIM, cudaMemcpyHostToDevice);
      } else {
        Volume<T, NDIM> tempVol;
        cudaMemcpy(
          lows, tempVol.lows, sizeof(T) * NDIM, cudaMemcpyHostToDevice);
        cudaMemcpy(
          highs, tempVol.highs, sizeof(T) * NDIM, cudaMemcpyHostToDevice);
      }
      //nvtxRangePop();
    }

    void
    RelErrClassify(int* activeRegions,
                   // double iter_estimate,
                   // double leaves_estimate,
                   // double finished_estimate,
                   // double finished_errorest,
                   size_t nregions,
                   // size_t total_nregions,
                   double epsrel,
                   // double epsabs,
                   int iteration
                   // bool estimateHasConverged
    )
    {
      // can't use dRegionsError to store newErrs, because sibling errors are
      // accessed
      size_t numBlocks =
        numRegions / BLOCK_SIZE + ((numRegions % BLOCK_SIZE) ? 1 : 0);
      if (iteration == 0) {
        return;
      }

      T* newErrs = 0;
      QuadDebug(
        Device.AllocateMemory((void**)&newErrs, sizeof(double) * numRegions));
      //printf("Refine Err Relerr classify %lu regions\n", numRegions);
      RefineError<double>
        <<<numBlocks, BLOCK_SIZE>>>(dRegionsIntegral,
                                    dRegionsError,
                                    dParentsIntegral,
                                    dParentsError,
                                    newErrs,
                                    activeRegions,
                                    numRegions,
                                    // total_nregions,
                                    // iter_estimate,
                                    // leaves_estimate,
                                    // finished_estimate,
                                    // finished_errorest,
                                    // partitionManager.queued_reg_estimate,
                                    // partitionManager.queued_reg_errorest,
                                    epsrel,
                                    // epsabs,
                                    // iteration,
                                    // estimateHasConverged,
                                    // lastErr,
                                    heuristicID);

      cudaDeviceSynchronize();
      CudaCheckError();

      QuadDebug(cudaMemcpy(dRegionsError,
                           newErrs,
                           sizeof(T) * numRegions,
                           cudaMemcpyDeviceToDevice));
      QuadDebug(cudaFree(newErrs));
    }

    std::string
    doubleToString(double val, int prec_level)
    {
      std::ostringstream out;
      out.precision(prec_level);
      out << std::fixed << val;
      return out.str();
    }

    bool
    sigDigitsSame(double x, double y, double z, int requiredDigits)
    {
      double third = abs(x);
      double second = abs(y);
      double first = abs(z);

      while (first < 1.) {
        first *= 10;
      }
      while (second < 1.) {
        second *= 10;
      }
      while (third < 1.) {
        third *= 10;
      }

      std::string second_to_last = doubleToString(third, 15);
      std::string last = doubleToString(second, 15);
      std::string current = doubleToString(first, 15);

      bool verdict = true;
      int sigDigits = 0;

      for (int i = 0; i < requiredDigits + 1 && sigDigits < requiredDigits &&
                      verdict == true;
           ++i) {
        verdict =
          current[i] == last[i] && last[i] == second_to_last[i] ? true : false;

        sigDigits += (verdict == true && current[i] != '.') ? 1 : 0;
      }
      return verdict;
    }

    //-----------------------------------------------------------------------------------------------------------
    void
    IterationAllocations(double*& dRegionsIntegral,
                         double*& dRegionsError,
                         double*& dParentsIntegral,
                         double*& dParentsError,
                         int*& activeRegions,
                         int*& subDividingDimension,
                         int iteration)
    {
      dRegionsError = nullptr, dRegionsIntegral = nullptr;

      QuadDebug(Device.AllocateMemory((void**)&dRegionsIntegral,
                                      sizeof(T) * numRegions));
      QuadDebug(
        Device.AllocateMemory((void**)&dRegionsError, sizeof(T) * numRegions));
      if (iteration == 0) {
        QuadDebug(Device.AllocateMemory((void**)&dParentsIntegral,
                                        sizeof(T) * numRegions));
        QuadDebug(Device.AllocateMemory((void**)&dParentsError,
                                        sizeof(T) * numRegions));
      }

      QuadDebug(Device.AllocateMemory((void**)&activeRegions,
                                      sizeof(int) * numRegions));
      QuadDebug(Device.AllocateMemory((void**)&subDividingDimension,
                                      sizeof(int) * numRegions));
    }

    double
    ComputeIterContribution(double* estimates)
    {
      // printf("performing reduction these %lu values\n", numRegions);
      // display<double, 1>(numRegions, estimates);
      thrust::device_ptr<double> wrapped_ptr =
        thrust::device_pointer_cast(estimates);
      return thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);
    }

    void
    FixErrorBudgetOverflow(int* activeRegions,
                           double& integral,
                           double& error,
                           double& iter_finished_estimate,
                           double& iter_finished_errorest,
                           double newLeavesEstimate,
                           double epsrel)
    {
      if (error > abs(newLeavesEstimate) * epsrel) {
        size_t numBlocks =
          numRegions / BLOCK_SIZE + ((numRegions % BLOCK_SIZE) ? 1 : 0);
        RevertFinishedStatus<<<numBlocks, BLOCK_SIZE>>>(activeRegions,
                                                        numRegions);
        // printf("Warning triggered, removing %.15e +- %.15e from finished
        // contributions\n", iter_finished_estimate, iter_finished_errorest);
        error -= iter_finished_errorest;
        integral -= iter_finished_estimate;
        // printf("Reverted estimates to %.15f +- %.15f\n", integral, error);
        iter_finished_estimate = 0.;
        iter_finished_errorest = 0.;
        cudaDeviceSynchronize();
      }
    }

    void
    ComputeFinishedEstimates(double& iter_finished_estimate,
                             double& iter_finished_errorest,
                             double* dRegionsIntegral,
                             double iter_estimate,
                             double* dRegionsError,
                             double iter_errorest,
                             int* activeRegions)
    {
      if (phase2 == true && numRegions > first_phase_maxregions)
        return;
      thrust::device_ptr<int> wrapped_mask;
      //printf("1st Inner dot product %lu regions\n", numRegions);
      //nvtxRangePush("Inner Product 1");
      wrapped_mask = thrust::device_pointer_cast(activeRegions);
      
      thrust::device_ptr<T> wrapped_ptr =
        thrust::device_pointer_cast(dRegionsIntegral);
      iter_finished_estimate =
        iter_estimate - thrust::inner_product(thrust::device,
                                              wrapped_ptr,
                                              wrapped_ptr + numRegions,
                                              wrapped_mask,
                                              0.);
      //nvtxRangePop();
      //printf("2nd Inner dot product %lu regions\n", numRegions);
      //nvtxRangePush("Inner Product 2");
      wrapped_ptr = thrust::device_pointer_cast(dRegionsError);
      iter_finished_errorest =
        iter_errorest - thrust::inner_product(thrust::device,
                                              wrapped_ptr,
                                              wrapped_ptr + numRegions,
                                              wrapped_mask,
                                              0.);
      //nvtxRangePop();
    }

    void
    CheckZeroNumRegionsTermination(double& integral,
                                   double& error,
                                   double leaves_estimate,
                                   double leaves_errorest)
    {
      if (numRegions == 0 && partitionManager.GetNRegions() == 0) {
        integral = leaves_estimate;
        error = leaves_errorest;
      }
    }

    bool
    CheckTerminationCondition(double leaves_estimate,
                              double leaves_errorest,
                              double& integral,
                              double& error,
                              size_t& nregions,
                              double epsrel,
                              double epsabs,
                              int iteration,
                              int* activeRegions,
                              int* subDividingDimension)
    {
      // if(mustFinish == true)
      // printf("Must finish triggered\n");
      if ((iteration != 0 &&
           leaves_errorest <= MaxErr(leaves_estimate, epsrel, epsabs)) ||
          mustFinish) {

        integral = leaves_estimate;
        error = leaves_errorest;
        nregions += numRegions + partitionManager.NumRegionsStored();

        if (leaves_errorest <= MaxErr(leaves_estimate, epsrel, epsabs))
          fail = 0;
        else
          fail = 1;
        // printf("About to terminate with fail:%i\n", fail);
        numRegions = 0;
        QuadDebug(cudaFree(activeRegions));
        QuadDebug(cudaFree(subDividingDimension));
        return true;
      } else
        return false;
    }

    template <typename IntegT>
    void
    FirstPhaseIteration(IntegT* d_integrand,
                        T epsrel,
                        T epsabs,
                        T& integral,
                        T& error,
                        size_t& nregions,
                        size_t& nFinishedRegions,
                        size_t& neval,
                        T*& dParentsIntegral,
                        T*& dParentsError,
                        int iteration,
                        Volume<T, NDIM>* vol,
                        int last_iteration = 0)
    {
      size_t numThreads = BLOCK_SIZE;
      size_t numBlocks = numRegions;
      // printf("Amount of free memory at start of iteration:%lu\n",
      // Device.GetAmountFreeMem());

      //nvtxRangePush("Iteration Allocations");
      dRegionsError = nullptr, dRegionsIntegral = nullptr;
      int *activeRegions = 0, *subDividingDimension = 0;
      IterationAllocations(dRegionsIntegral,
                           dRegionsError,
                           dParentsIntegral,
                           dParentsError,
                           activeRegions,
                           subDividingDimension,
                           iteration);
      CudaCheckError();
      //nvtxRangePop();

      /*if(iteration == 14){
          printf("about to display\n");
          display<double, 2>(numRegions*NDIM, dRegions, dRegionsLength);
      }*/

      //nvtxRangePush("INTEGRATE_GPU_PHASE1");
      //printf("Launching for %lu regions\n", numBlocks);
      INTEGRATE_GPU_PHASE1<IntegT, T, NDIM, BLOCK_SIZE>
        <<<numBlocks, numThreads /*, NDIM * sizeof(GlobalBounds)*/>>>(
          d_integrand,
          dRegions,
          dRegionsLength,
          numRegions,
          dRegionsIntegral,
          dRegionsError,
          activeRegions,
          subDividingDimension,
          epsrel,
          epsabs,
          constMem,
          rule.GET_FEVAL(),
          rule.GET_NSETS(),
          lows,
          highs,
          iteration,
          depthBeingProcessed,
          generators);

      neval += numRegions * fEvalPerRegion;
      cudaDeviceSynchronize();
      CudaCheckError();
      //nvtxRangePop();
      //printf("Reduction 1 %lu regions\n", numRegions);
      //nvtxRangePush("Reduction 1");
      double iter_estimate = ComputeIterContribution(dRegionsIntegral);
      //nvtxRangePop();
      double leaves_estimate =
        partitionManager.queued_reg_estimate + integral + iter_estimate;
      nvtxRangePush("Rel Error Classify");
      RelErrClassify(
        activeRegions, nregions, epsrel, /*depthBeingProcessed*/ iteration);
      nvtxRangePop();
      //printf("Reduction 2 %lu regions\n", numRegions);
      double iter_finished_estimate = 0, iter_finished_errorest = 0;
      //nvtxRangePush("Reduction 2");
      double iter_errorest = ComputeIterContribution(dRegionsError);
      //nvtxRangePop();
      double leaves_errorest =
        partitionManager.queued_reg_errorest + error + iter_errorest;

      ComputeFinishedEstimates(iter_finished_estimate,
                               iter_finished_errorest,
                               dRegionsIntegral,
                               iter_estimate,
                               dRegionsError,
                               iter_errorest,
                               activeRegions);
      integral += iter_finished_estimate;
      error += iter_finished_errorest;
      // printf("%i, %.15f, %.15f, %.15f, %.15f numRegions:%lu\n", iteration,
      // leaves_estimate, leaves_errorest, iter_estimate, iter_errorest,
      // numRegions); printf("%i, iter estimates: %.15f, %.15f (%.15e +- %.15e),
      // numRegions:%lu\n", iteration, iter_estimate, iter_errorest,
      // iter_estimate, iter_errorest, numRegions);
      // printf("-----------------------\n");
      Phase_I_PrintFile(vol,
                        numRegions,
                        activeRegions,
                        leaves_estimate,
                        leaves_errorest,
                        iter_estimate,
                        iter_errorest,
                        iter_finished_estimate,
                        iter_finished_errorest,
                        partitionManager.queued_reg_estimate,
                        partitionManager.queued_reg_errorest,
                        partitionManager.GetNRegions(),
                        epsrel,
                        epsabs,
                        iteration);

      FixErrorBudgetOverflow(activeRegions,
                             integral,
                             error,
                             iter_finished_estimate,
                             iter_finished_errorest,
                             leaves_estimate,
                             epsrel);
      // printf("%i, iter estimates: %.15f, %.15f (%.15e +- %.15e),
      // numRegions:%lu\n", iteration, iter_estimate, iter_errorest,
      // iter_estimate, iter_errorest, numRegions);
      if (/*GetGPUMemNeededForNextIteration_CallBeforeSplit() >=
       Device.GetAmountFreeMem() && mustFinish == true && */
          CheckTerminationCondition(leaves_estimate,
                                    leaves_errorest,
                                    integral,
                                    error,
                                    nregions,
                                    epsrel,
                                    epsabs,
                                    iteration,
                                    activeRegions,
                                    subDividingDimension))
        return;

      nvtxRangePush("HS CLASSIFY");
      HSClassify(dRegionsIntegral,
                 dRegionsError,
                 subDividingDimension,
                 dParentsIntegral,
                 dParentsError,
                 activeRegions,
                 integral,
                 error,
                 nregions,
                 iter_estimate,
                 iter_errorest,
                 iter_finished_estimate,
                 iter_finished_errorest,
                 leaves_estimate,
                 leaves_errorest,
                 epsrel,
                 epsabs,
                 iteration);
      nvtxRangePop();
      if (CheckTerminationCondition(leaves_estimate,
                                    leaves_errorest,
                                    integral,
                                    error,
                                    nregions,
                                    epsrel,
                                    epsabs,
                                    iteration,
                                    activeRegions,
                                    subDividingDimension))
        return;
      
      // 
      if (iteration < 700 && fail == 1 &&
          (phase2 == false || numRegions <= first_phase_maxregions)) {
        nvtxRangePush("GenerateActiveIntervals");
        size_t numInActiveIntervals =
          GenerateActiveIntervals(activeRegions,
                                  subDividingDimension,
                                  dRegionsIntegral,
                                  dRegionsError,
                                  dParentsIntegral,
                                  dParentsError);
        CheckZeroNumRegionsTermination(
          integral, error, leaves_estimate, leaves_errorest);
        depthBeingProcessed++;
        nregions += numInActiveIntervals;
        nFinishedRegions += numInActiveIntervals;
      } else {
        nregions += numRegions + partitionManager.GetNRegions();

        phase2Ready = true;
      }
      nvtxRangePop();
      QuadDebug(cudaFree(activeRegions));
      QuadDebug(cudaFree(subDividingDimension));
      
    }

    template <typename IntegT>
    bool
    IntegrateFirstPhase(IntegT* d_integrand,
                        T epsrel,
                        T epsabs,
                        T& integral,
                        T& error,
                        size_t& nregions,
                        size_t& nFinishedRegions,
                        size_t& neval,
                        Volume<T, NDIM>* vol = nullptr)
    {
      QuadDebug(Device.AllocateMemory((void**)&generators,
                                      sizeof(double) * NDIM * fEvalPerRegion));
      ComputeGenerators<NDIM>
        <<<1, BLOCK_SIZE>>>(generators, fEvalPerRegion, constMem);
      cudaDeviceSynchronize();
      CudaCheckError();

      AllocVolArrays(vol);
      CudaCheckError();
      PrintOutfileHeaders();
      int lastIteration = 0;
      int iteration = 0;
      fail = 1;

      for (iteration = 0; iteration < 700 && phase2Ready == false &&
                          fail == 1 && mustFinish == false;
           iteration++) {
        FirstPhaseIteration<IntegT>(d_integrand,
                                    epsrel,
                                    epsabs,
                                    integral,
                                    error,
                                    nregions,
                                    nFinishedRegions,
                                    neval,
                                    dParentsIntegral,
                                    dParentsError,
                                    iteration,
                                    vol,
                                    lastIteration);
        QuadDebug(cudaFree(dRegionsError));
        QuadDebug(cudaFree(dRegionsIntegral));
      }

      CudaCheckError();

      StringstreamToFile(finishedOutfile.str(), phase1out.str(), outLevel);
      QuadDebug(Device.ReleaseMemory(dRegions));
      QuadDebug(Device.ReleaseMemory(dRegionsLength));
      QuadDebug(Device.ReleaseMemory(dParentsIntegral));
      QuadDebug(Device.ReleaseMemory(dParentsError));
      QuadDebug(Device.ReleaseMemory(lows));
      QuadDebug(Device.ReleaseMemory(highs));
      QuadDebug(cudaFree(generators));

      bool convergence = false;
      convergence = error <= MaxErr(integral, epsrel, epsabs);
      return !convergence;
    }
  };

}
#endif
