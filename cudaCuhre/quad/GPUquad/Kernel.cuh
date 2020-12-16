#ifndef CUDACUHRE_QUAD_GPUQUAD_KERNEL_CUH
#define CUDACUHRE_QUAD_GPUQUAD_KERNEL_CUH

#include "../util/Volume.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "PartitionManager.cuh"
#include "Phases.cuh"
#include "Rule.cuh"

#include <cuda.h>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <thrust/transform_reduce.h>

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

  template <typename T>
  __global__ void
  PrintcuArray(T* array, int size)
  {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int i = 0; i < size; i++) {
        // if(i<10)
        printf("array[%i]:%.12f\n", i, array[i]);
        printf("array[%i]:%.12f\n", i, array[i]);
      }
    }
  }

  template <typename T>
  __global__ void
  PrintcuArray(T* array, T* array2, int size)
  {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int i = 0; i < size; i++)
        printf("array[%i]:%.12f - %.12f\n", i, array[i], array[i] + array2[i]);
    }
  }

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
        dRegionsIntegral[tid + numRegions];
      dRegionsParentError[interval_index] = dRegionsError[tid + numRegions];

      dRegionsParentIntegral[interval_index + newNumRegions] =
        dRegionsIntegral[tid + numRegions];
      dRegionsParentError[interval_index + newNumRegions] =
        dRegionsError[tid + numRegions];

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

    bool estimateHasConverged;
    int Final;
    int phase_I_type;
    int heuristicID;
    int fail; // 0 for satisfying ratio, 1 for not satisfying ratio, 2 for
              // running out of bad regions
    int phase2_failedblocks;
    T lastErr;
    T lastAvg;
    T secondTolastAvg;

    double errorest_change;
    double estimate_change;

    T* dlastErr;
    T* dlastAvg;
    int KEY, VERBOSE, outLevel;
    size_t numRegions, h_numRegions, numFunctionEvaluations, numInActiveRegions;
    size_t fEvalPerRegion;
    int first_phase_maxregions;
    int max_globalpool_size;
    size_t host_current_list_id;

    const size_t numHostPartitions = 4;
    size_t partionSize[4];
    double partionContributionsIntegral[4];
    double partionContributionsError[4];

    int nextAvailRegionID;
    int nextAvailParentID;
    size_t* parentIDs;

    HostMemory<T> Host;
    DeviceMemory<T> Device;
    Rule<T> rule;
    Structures<T> constMem;
    int NUM_DEVICES;
    // Debug Msg
    char msg[256];

    std::ostream& log;

  public:
    T weightsum, avgsum, guess, chisq, chisum, chisqsum;

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
    SetHeuristicID(int id){
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
      QuadDebug(cudaMemcpy(
        temp, array, sizeof(T) * copy_size, cudaMemcpyDeviceToDevice));
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

    Kernel(std::ostream& logerr = std::cout) : log(logerr)
    {
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

      /*QuadDebug(Device.AllocateMemory((void**)&gRegionPool,
                                      sizeof(Region<NDIM>) *
                                        first_phase_maxregions * 2 *
                                        max_globalpool_size));*/

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
      QuadDebug(Device.ReleaseMemory(dRegions));
      QuadDebug(Device.ReleaseMemory(dRegionsLength));

      QuadDebug(Device.ReleaseMemory(dParentsIntegral));
      QuadDebug(Device.ReleaseMemory(dParentsError));

      QuadDebug(Device.ReleaseMemory(lows));
      QuadDebug(Device.ReleaseMemory(highs));

      //QuadDebug(Device.ReleaseMemory(gRegionPool));
      Host.ReleaseMemory(curr_hRegions);
      Host.ReleaseMemory(curr_hRegionsLength);

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
      QuadDebug(cudaDeviceSynchronize());
    }

    void
    StringstreamToFile(std::string per_iteration,
                       std::string per_region,
                       int verbosity)
    {
      switch (verbosity) {
        case 1:
          PrintToFile(per_iteration, "Per_iteration.csv");
          break;
        case 2:
          PrintToFile(per_iteration, "Per_iteration.csv");
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
        finishedOutfile << "iteration, tot.est, tot.err, it.nreg, it.est, "
                           "it.err,  it.fin.est, it.fin.err, it.fin.nreg, "
                           "uneval.par.est, uneval.par.err, uneval.nreg\n";

      finishedOutfile << std::setprecision(17) << std::scientific << iteration
                      << "," << leaves_estimate << "," << leaves_errorest << ","
                      << iter_nregions << "," << iter_estimate << ","
                      << iter_errorest << "," << iter_finished_estimate << ","
                      << iter_finished_errorest << "," << dnumInActiveRegions
                      << "," << queued_estimate << "," << queued_errorest << ","
                      << unevaluated_nregions << "\n";
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

      double sum_est = 0.;
      double sum_errorest = 0.;
      bool free_bounds_needed = false;

      size_t numActiveRegions = 0;
      int* h_activeRegions;
      h_activeRegions = (int*)malloc(sizeof(int) * iter_nregions);

      double* curr_hRegionsIntegral = nullptr;
      double* curr_hRegionsError = nullptr;
      double* curr_ParentsIntegral = nullptr;
      double* curr_ParentsError = nullptr;

      curr_hRegionsIntegral = (double*)malloc(sizeof(double) * iter_nregions);
      curr_hRegionsError = (double*)malloc(sizeof(double) * iter_nregions);
      curr_ParentsIntegral = (double*)malloc(sizeof(double) * iter_nregions);
      curr_ParentsError = (double*)malloc(sizeof(double) * iter_nregions);

      QuadDebug(cudaMemcpy(curr_hRegionsIntegral,
                           dRegionsIntegral + iter_nregions,
                           sizeof(double) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_hRegionsError,
                           dRegionsError + iter_nregions,
                           sizeof(double) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_ParentsIntegral,
                           dParentsIntegral,
                           sizeof(double) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_ParentsError,
                           dParentsError,
                           sizeof(double) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(h_activeRegions,
                           activeRegions,
                           sizeof(int) * iter_nregions,
                           cudaMemcpyDeviceToHost));

      CudaCheckError();
      if (curr_hRegions == nullptr && curr_hRegionsLength == nullptr) {
        curr_hRegions = (double*)malloc(sizeof(double) * iter_nregions * NDIM);
        curr_hRegionsLength =
          (double*)malloc(sizeof(double) * iter_nregions * NDIM);
        free_bounds_needed = true;
        QuadDebug(cudaMemcpy(curr_hRegions,
                             dRegions,
                             sizeof(double) * iter_nregions * NDIM,
                             cudaMemcpyDeviceToHost));
        QuadDebug(cudaMemcpy(curr_hRegionsLength,
                             dRegionsLength,
                             sizeof(double) * iter_nregions * NDIM,
                             cudaMemcpyDeviceToHost));
      }

      CudaCheckError();

      if (iteration == 0) {
        phase1out << "iteration, id, parentID, estimate, errorest, parEst, "
                     "parErr, active,";
        for (int i = 0; i < NDIM; i++) {
          std::string dim = std::to_string(i);
          phase1out << "dim" + dim + "low, dim" + dim + "high";
          if (i == NDIM - 1)
            phase1out << "\n";
          else
            phase1out << ",";
        }
      }
      
      for (int regnIndex = 0; regnIndex < iter_nregions; regnIndex++) {
        sum_est += curr_hRegionsIntegral[regnIndex];
        sum_errorest += curr_hRegionsError[regnIndex];

        int parentID = 0;

        if (iter_nregions > 1)
          parentID = regnIndex < iter_nregions / 2 ?
                       parentIDs[regnIndex] :
                       parentIDs[regnIndex - iter_nregions / 2];
        else
          parentID = -1;

        double ratio = curr_hRegionsError[regnIndex] /
                       (epsrel * abs(curr_hRegionsIntegral[regnIndex]));
        /*if(iteration == 24)*/ {
          phase1out << std::setprecision(17) << std::scientific << iteration
                    << "," << nextAvailRegionID + regnIndex << "," << parentID
                    << "," << curr_hRegionsIntegral[regnIndex] << ","
                    << curr_hRegionsError[regnIndex] << ","
                    << curr_ParentsIntegral[regnIndex] << ","
                    << curr_ParentsError[regnIndex] << ","
                    << h_activeRegions[regnIndex] << ",";
        }

        if (h_activeRegions[regnIndex] == 1)
          numActiveRegions++;

        for (int dim = 0; dim < NDIM; dim++) {
          double low =
            ScaleValue(curr_hRegions[dim * iter_nregions + regnIndex],
                       vol->lows[dim],
                       vol->highs[dim]);

          // could this be wrong? doesn't seem so
          double high =
            ScaleValue(curr_hRegions[dim * iter_nregions + regnIndex] +
                         curr_hRegionsLength[dim * iter_nregions + regnIndex],
                       vol->lows[dim],
                       vol->highs[dim]);

          if (dim == NDIM - 1 /*&& iteration == 24*/) {
            phase1out << std::setprecision(17) << std::scientific << low << ","
                      << high << "\n";
          } else /*if(iteration == 24)*/ {
            phase1out << std::setprecision(17) << std::scientific << low << ","
                      << high << ",";
          }
        }
        // if(iteration == 24)
        phase1out << "\n";
      }

      if (parentIDs != nullptr)
        delete[] parentIDs;

      parentIDs = new size_t[numActiveRegions];

      size_t nextActiveRegion = 0;
      for (int i = 0; i < iter_nregions; i++) {
        if (h_activeRegions[i] == 1) {
          parentIDs[nextActiveRegion] = nextAvailRegionID + i;
          nextActiveRegion++;
        }
      }

      nextAvailRegionID += iter_nregions;

      free(curr_hRegionsError);
      free(curr_hRegionsIntegral);
      free(h_activeRegions);

      if (free_bounds_needed) {
        free(curr_hRegions);
        free(curr_hRegionsLength);
        curr_hRegions = nullptr;
        curr_hRegionsLength = nullptr;
      }
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

    template <class K>
    void
    displayDuplicate(K* array1,
                     K* array2,
                     size_t size,
                     std::string msg = std::string())
    {
      K* tmp1 = (K*)malloc(sizeof(K) * size * 2);
      K* tmp2 = (K*)malloc(sizeof(K) * size * 2);
      cudaMemcpy(tmp1, array1, sizeof(K) * size * 2, cudaMemcpyDeviceToHost);
      cudaMemcpy(tmp2, array2, sizeof(K) * size * 2, cudaMemcpyDeviceToHost);

      std::cout << msg << "\n";

      for (int i = 0; i < size * 2; ++i)
        printf("%i, %.20f, %.20f\n", i, (T)tmp1[i], (T)tmp2[i]);

      free(tmp1);
      free(tmp2);
    }

    template <class K>
    void
    displayIfNotZero(K* array1, K* array2, size_t size)
    {
      K* tmp1 = (K*)malloc(sizeof(K) * size);
      K* tmp2 = (K*)malloc(sizeof(K) * size);
      cudaMemcpy(tmp1, array1, sizeof(K) * size, cudaMemcpyDeviceToHost);
      cudaMemcpy(tmp2, array2, sizeof(K) * size, cudaMemcpyDeviceToHost);
      for (int i = 0; i < size; ++i) {
        // std::cout << tmp[i] << std::endl;
        if ((T)tmp2[i] == 0. || (T)tmp1[i] == 0.) {
          printf("%i, %.20f, %.20f, r:%f\n",
                 i,
                 (T)tmp1[i],
                 (T)tmp2[i],
                 tmp2[i] / MaxErr(tmp1[i], 1e-3, 1.0e-20));
        }
      }

      free(tmp1);
      free(tmp2);
    }

    template <class K>
    void
    displayIfNotZero(K* array, size_t size)
    {
      K* tmp = (K*)malloc(sizeof(K) * size);
      cudaMemcpy(tmp, array, sizeof(K) * size, cudaMemcpyDeviceToHost);
      for (int i = 0; i < size; ++i) {
        // std::cout << tmp[i] << std::endl;
        if ((T)tmp[i] == 0.)
          printf("%i, %.20f ||\n", i, (T)tmp[i]);
      }
    }

    template <class K>
    void
    display(K* array,
            size_t size,
            std::string msg = std::string(),
            int specific_index = -1,
            int specific_index2 = -1)
    {
      K* tmp = (K*)malloc(sizeof(K) * size);
      cudaMemcpy(tmp, array, sizeof(K) * size, cudaMemcpyDeviceToHost);

      if (specific_index != -1) {
        printf("About to display\n");
        printf("display, %i, %e\n", specific_index, (T)tmp[specific_index]);

        if (specific_index2 != -1)
          printf("display, %i, %e\n", specific_index2, (T)tmp[specific_index2]);
        return;
      }

      for (size_t i = 0; i < size; ++i) {
        printf("%s, %i, %e\n", msg.c_str(), i, (T)tmp[i]);
      }

      free(tmp);
    }

    template <class K>
    void
    display(K* array, size_t size, std::string filename, std::string header)
    {
      std::stringstream outfile;
      outfile << header << std::endl;
      K* tmp = (K*)malloc(sizeof(K) * size);
      cudaMemcpy(tmp, array, sizeof(K) * size, cudaMemcpyDeviceToHost);
      for (int i = 0; i < size; ++i) {
        outfile << tmp[i] << std::endl;
        // printf("%.20lf \n", (T)tmp[i]);
      }
      PrintToFile(outfile.str(), filename);
    }

    template <class K>
    void
    display(K* array1,
            K* array2,
            T optional,
            size_t size,
            K (*func)(K, K, T),
            std::string filename,
            std::string header)
    {
      std::stringstream outfile;
      K* tmp1 = (K*)malloc(sizeof(K) * size);
      K* tmp2 = (K*)malloc(sizeof(K) * size);

      cudaMemcpy(tmp1, array1, sizeof(K) * size, cudaMemcpyDeviceToHost);
      cudaMemcpy(tmp2, array2, sizeof(K) * size, cudaMemcpyDeviceToHost);

      outfile << header << std::endl;

      for (int i = 0; i < size; ++i)
        outfile << tmp1[i] << "," << tmp2[i] << ","
                << func(tmp1[i], tmp2[i], optional) << std::endl;

      std::string outputS = outfile.str();
      PrintToFile(outputS, filename);
    }

    template <class K>
    void
    display(K* array1, K* array2, size_t size, std::string msg = std::string())
    {
      std::stringstream outfile;
      K* tmp1 = new K[size]; //(K*)malloc(sizeof(K) * size);
      K* tmp2 = new K[size]; //(K*)malloc(sizeof(K) * size);

      cudaMemcpy(tmp1, array1, sizeof(K) * size, cudaMemcpyDeviceToHost);
      cudaMemcpy(tmp2, array2, sizeof(K) * size, cudaMemcpyDeviceToHost);

      std::cout << msg << "\n";
      for (int i = 0; i < size; ++i)
        printf("display, %e, %e\n", i, (T)tmp1[i], (T)tmp2[i]);
    }

    template <class K>
    void
    display(K* array1,
            K* array2,
            int* condition,
            size_t size,
            std::string msg = std::string())
    {
      std::stringstream outfile;
      K* tmp1 = new K[size]; //(K*)malloc(sizeof(K) * size);
      K* tmp2 = new K[size]; //(K*)malloc(sizeof(K) * size);
      int* tmp3 = new int[size];

      cudaMemcpy(tmp1, array1, sizeof(K) * size, cudaMemcpyDeviceToHost);
      cudaMemcpy(tmp2, array2, sizeof(K) * size, cudaMemcpyDeviceToHost);
      cudaMemcpy(tmp3, condition, sizeof(int) * size, cudaMemcpyDeviceToHost);

      std::cout << msg << "\n";
      for (int i = 0; i < size; ++i) {
        if (tmp3[i] == 0)
          printf("display, %e, %e\n", i, (T)tmp1[i], (T)tmp2[i]);
      }
    }

    size_t
    GetGPUMemNeededForNextIteration()
    {
      // numRegions implies the number of regions processed in next iteration
      // so it's supposed to be called only after GenerateIntervals has been
      // called
      int numOfDivisionOnDimension = 2;
      size_t nextIter_dParentsIntegral_size = numRegions * 2;
      size_t nextIter_dParentsError_size = numRegions * 2;
      size_t nextIter_dRegions_size = numRegions * NDIM;
      size_t nextIter_dRegionsLength_size = numRegions * NDIM;
      size_t nextIter_dRegionsIntegral_size =
        numRegions * 2; // is this really times two?
      size_t nextIter_dRegionsError_size = numRegions * 2;
      size_t nextIter_newActiveRegions_size = numRegions * NDIM * numOfDivisionOnDimension;
      size_t nextIter_newActiveRegionsLength_size = numRegions * NDIM * numOfDivisionOnDimension;

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
      // size_t numOfDivisionPerRegionPerDimension = 4;
      // if(NDIM == 5 )numOfDivisionPerRegionPerDimension = 2;
      // if(NDIM == 6 )numOfDivisionPerRegionPerDimension = 2;
      // if(NDIM == 7 )numOfDivisionPerRegionPerDimension = 2;
      // if(NDIM > 7 )numOfDivisionPerRegionPerDimension = 2;
      // if(NDIM > 10 )numOfDivisionPerRegionPerDimension = 1;

      size_t numOfDivisionPerRegionPerDimension = 1;

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
      free(curr_hRegions);
      free(curr_hRegionsLength);
      curr_hRegions = nullptr;
      curr_hRegionsLength = nullptr;
    }

    size_t
    GenerateActiveIntervals(int* activeRegions,
                            int* subDividingDimension,
                            T* dRegionsIntegral,
                            T* dRegionsError,
                            T*& dParentsIntegral,
                            T*& dParentsError)
    {

      int* scannedArray = 0; // de-allocated at the end of this function
      QuadDebug(
        Device.AllocateMemory((void**)&scannedArray, sizeof(int) * numRegions));

      thrust::device_ptr<int> d_ptr =
        thrust::device_pointer_cast(activeRegions);
      thrust::device_ptr<int> scan_ptr =
        thrust::device_pointer_cast(scannedArray);
      thrust::exclusive_scan(d_ptr, d_ptr + numRegions, scan_ptr);

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
        
      numInActiveRegions = numRegions - numActiveRegions;
      //printf("Bad Reginos %lu/%lu\n", numActiveRegions, numRegions);
      if (outLevel >= 4)
        out4 << numActiveRegions << "," << numRegions << std::endl;

      if (numActiveRegions > 0) {

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

        ExpandcuArray(dParentsIntegral, numRegions, numActiveRegions * 2);
        CudaCheckError();
        ExpandcuArray(dParentsError, numRegions, numActiveRegions * 2);
        // printf("After expansion dParents size:%lu\n", numActiveRegions * 4);
        CudaCheckError();
        size_t numThreads = BLOCK_SIZE;
        size_t numBlocks =
          numRegions / numThreads + ((numRegions % numThreads) ? 1 : 0);

        cudaDeviceSynchronize();
        // printf("Populating dParents with the results of dRegionsIntegral\n");
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
        T *genRegions = 0, *genRegionsLength = 0;
        numBlocks = numActiveRegions / numThreads +
                    ((numActiveRegions % numThreads) ? 1 : 0);

        // IDEA can use expandcuArray(
       
        QuadDebug(cudaMalloc((void**)&genRegions,
                             sizeof(T) * numActiveRegions * NDIM *
                               numOfDivisionOnDimension));
        QuadDebug(cudaMalloc((void**)&genRegionsLength,
                             sizeof(T) * numActiveRegions * NDIM *
                               numOfDivisionOnDimension));
        CudaCheckError();

        QuadDebug(Device.ReleaseMemory(dRegions));
        QuadDebug(Device.ReleaseMemory(dRegionsLength));
        QuadDebug(Device.ReleaseMemory(scannedArray));

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
        cudaDeviceSynchronize();

      } else {
        numRegions = 0;
      }

      return numInActiveRegions;
    }

    void
    AllocVolArrays(Volume<T, NDIM>* vol)
    {
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
    }

    void
    hRefineError(int* activeRegions,
                 double iter_estimate,
                 double leaves_estimate,
                 double finished_estimate,
                 double finished_errorest,
                 size_t nregions,
                 size_t total_nregions,
                 double epsrel,
                 double epsabs,
                 int iteration,
                 bool estimateHasConverged)
    {
      // can't use dRegionsError to store newErrs, because sibling errors are
      // accessed
      size_t numBlocks = numRegions / BLOCK_SIZE + ((numRegions % BLOCK_SIZE) ? 1 : 0);
      if (numRegions == startRegions)
        return;

      T* newErrs = 0;
      QuadDebug(Device.AllocateMemory((void**)&newErrs,
                                      sizeof(double) * numRegions * 2));
        
      RefineError<double>
        <<<numBlocks, BLOCK_SIZE>>>(dRegionsIntegral,
                                     dRegionsError,
                                     dParentsIntegral,
                                     dParentsError,
                                     newErrs,
                                     activeRegions,
                                     numRegions,
                                     total_nregions,
                                     iter_estimate,
                                     leaves_estimate,
                                     finished_estimate,
                                     finished_errorest,
                                     partitionManager.queued_reg_errorest,
                                     epsrel,
                                     epsabs,
                                     iteration,
                                     estimateHasConverged,
                                     lastErr,
                                     heuristicID);

      cudaDeviceSynchronize();
      CudaCheckError();
      QuadDebug(cudaMemcpy(dRegionsError,
                           newErrs,
                           sizeof(T) * numRegions * 2,
                           cudaMemcpyDeviceToDevice));

      QuadDebug(cudaFree(newErrs));
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

      dRegionsError = nullptr, dRegionsIntegral = nullptr;

      QuadDebug(Device.AllocateMemory((void**)&dRegionsIntegral,
                                      sizeof(T) * numRegions * 2));
      QuadDebug(Device.AllocateMemory((void**)&dRegionsError,
                                      sizeof(T) * numRegions * 2));

      // move this to constructor, or init function
      if (numRegions == startRegions && error == 0) {
        QuadDebug(Device.AllocateMemory((void**)&dParentsIntegral,
                                        sizeof(T) * numRegions));
        QuadDebug(Device.AllocateMemory((void**)&dParentsError,
                                        sizeof(T) * numRegions));
      }

      int *activeRegions = 0, *subDividingDimension = 0;

      QuadDebug(Device.AllocateMemory((void**)&activeRegions,
                                      sizeof(int) * numRegions));
      QuadDebug(Device.AllocateMemory((void**)&subDividingDimension,
                                      sizeof(int) * numRegions));
      
      
      //cudaMemset(activeRegions, 1, numRegions*sizeof(int));
     
      INTEGRATE_GPU_PHASE1<IntegT, T, NDIM>
        <<<numBlocks, numThreads, NDIM * sizeof(GlobalBounds)>>>(
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
          depthBeingProcessed);

      cudaDeviceSynchronize();
      CudaCheckError();
      //int starting = 1;
      
      //
      
      neval += numRegions * fEvalPerRegion;
      thrust::device_ptr<T> wrapped_ptr;
      wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral + numRegions);

      double iter_estimate =
        thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);
      double leaves_estimate =
        partitionManager.queued_reg_estimate + integral + iter_estimate;
      estimate_change = abs(leaves_estimate - lastAvg);
    
      auto EstimateTrustWorthy = [iteration,
                                  lastAvg = abs(this->lastAvg),
                                  secondTolastAvg = abs(this->secondTolastAvg),
                                  leaves_estimate,
                                  epsrel]() {
        double third = abs(secondTolastAvg);
        double second = abs(lastAvg);
        double first = abs(leaves_estimate);
        int requiredDigits = ceil(log10(1 / epsrel));
        
        while (first / (10 * requiredDigits) < 1.) {
          first *= 10;
        }
        while (second / (10 * requiredDigits) < 1.) {
          second *= 10;
        }
        while (third / (10 * requiredDigits) < 1.) {
          third *= 10;
        }

        // i'm not relying on identifying individual zeros and '.' because
        // of implicit roundung that occurs on  to_string double parameters
        std::string second_to_last = std::to_string(third);
        std::string last = std::to_string(second);
        std::string current = std::to_string(first);

        double verdict = true;
        for (int i = 0; i < requiredDigits && verdict == true; ++i) {
          verdict = current[i] == last[i] && last[i] == second_to_last[i] ?
                      true :
                      false;
        }

        return verdict;
      };

      estimateHasConverged =
        estimateHasConverged == false ?
          (iteration >= 2 ? EstimateTrustWorthy() : false) :
          true;
      
      if(estimateHasConverged && error > leaves_estimate*epsrel)
        printf("it:%iTHREW AWAY TOO MUCH\n", iteration);
      
      auto GetNewEpsabs = [iter_estimate, 
                           error, 
                           leaves_estimate, 
                           estimateHasConverged=this->estimateHasConverged, 
                           numRegions=this->numRegions, 
                           epsrel, 
                           queued_errorest=this->partitionManager.queued_reg_errorest](){
                               
        size_t rough_region_cap = 1e9;
        double GlobErrTarget = fabs(leaves_estimate) * epsrel;
        double remainErrRoom = GlobErrTarget - error - queued_errorest;
        
        GlobErrTarget/rough_region_cap;
       //epsabs = estimateHasConverged ? (leaves_errorest-finished_errorest) estimated_soft_region_cap
       //printf("new epsabs:%e || %e\n", GlobErrTarget/rough_region_cap, remainErrRoom/rough_region_cap);
      };
      
      GetNewEpsabs();
      //printf("it:%i\t", iteration);
      hRefineError(activeRegions,
                   iter_estimate,
                   leaves_estimate,
                   integral,
                   error,
                   nregions,
                   nregions + numRegions,
                   epsrel,
                   epsabs,
                   depthBeingProcessed,//iteration, //change with depthBeingProcessed
                   estimateHasConverged);

      T temp_integral = 0;
      T temp_error = 0;

      if (last_iteration == 1) {
        // why is this used
        temp_integral = integral;
        temp_error = error;
      }

      wrapped_ptr = thrust::device_pointer_cast(dRegionsError + numRegions);
      double iter_errorest =
        thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);
      double leaves_errorest =
        partitionManager.queued_reg_errorest + error + iter_errorest;
      errorest_change = abs(leaves_errorest - lastErr);
      //printf("TEMP RESULT:%e +- %e\n", leaves_estimate, leaves_errorest);
      wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral);
      double iter_finished_estimate =
        thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);
      integral = integral + iter_finished_estimate;

      wrapped_ptr = thrust::device_pointer_cast(dRegionsError);
      double iter_finished_errorest =
        thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);
      error = error + iter_finished_errorest;

      /*printf("it:%i, est:%e, errorest:%e,  it_est:%e it_errorest:%e, fest:%e, ferrorest:%e , qerrorest:%e, qerrorest:%e, numRegions:%lu minIterReached:%i\n", iteration,
       leaves_estimate, 
       leaves_errorest, 
       iter_estimate,
       iter_errorest,
       integral,
       error,
       partitionManager.queued_reg_estimate,
       partitionManager.queued_reg_errorest,
       numRegions,
       estimateHasConverged);*/
       
      if (last_iteration != 1)
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

      if (Final == 0) {
        double w = numRegions * 1 /
                   fmax(leaves_errorest * leaves_errorest, ldexp(1., -104));
        weightsum += w; // adapted by Ioannis
        avgsum += w * leaves_estimate;
        double sigsq = 1 / weightsum;
        lastAvg = sigsq * avgsum;
        lastErr = sqrt(sigsq);
      } else {
        secondTolastAvg = lastAvg;
        lastAvg = leaves_estimate;
        lastErr = leaves_errorest;
      }

      if (iteration != 0 && (lastErr <= MaxErr(lastAvg, epsrel, epsabs)) &&
          GLOBAL_ERROR) {
        printf("Found answer naturally\n");
        integral = lastAvg;
        error = lastErr;
        QuadDebug(cudaFree(activeRegions));
        QuadDebug(cudaFree(subDividingDimension));
        //nFinishedRegions = nregions;
        nregions += numRegions + partitionManager.NumRegionsStored();
        fail = 0;
        numRegions = 0;

        return;
      } else if (last_iteration == 1) {
        integral = temp_integral;
        error = temp_error;
      }

      // double freeMem = Device.GetFreeMemPercentage();
      // if (numRegions <= first_phase_maxregions*8 && fail == 1) {
      // if (numRegions <= first_phase_maxregions && fail == 1) {

      if (iteration < 300 && fail == 1) {
        //  if (numRegions <= first_phase_maxregions && fail == 1) {
        size_t numInActiveIntervals =
          GenerateActiveIntervals(activeRegions,
                                  subDividingDimension,
                                  dRegionsIntegral,
                                  dRegionsError,
                                  dParentsIntegral,
                                  dParentsError);
        depthBeingProcessed++;
        nregions += numInActiveIntervals;
        nFinishedRegions += numInActiveIntervals;
        
        bool NotEnoughMem =
          GetGPUMemNeededForNextIteration() >= Device.GetAmountFreeMem();
        
        bool NoRegionsButPartitionsLeft =
          (numRegions == 0 && !partitionManager.Empty());
          
        if ((NotEnoughMem || NoRegionsButPartitionsLeft /*|| numRegions >= 4194304*/) && fail == 1) {
          //printf("Saving to Host Partition with %lu regions\n", numRegions);
          Partition<NDIM> currentPartition;
          currentPartition.ShallowCopy(dRegions,
                                       dRegionsLength,
                                       dParentsIntegral,
                                       dParentsError,
                                       numRegions,
                                       depthBeingProcessed);
          partitionManager.LoadNextActivePartition(
            currentPartition); // storeAndGetNextPartition

          // interface again with original implementation, to be changed
          dRegions = currentPartition.regions;
          dRegionsLength = currentPartition.regionsLength;
          dParentsIntegral = currentPartition.parentsIntegral;
          dParentsError = currentPartition.parentsError;
          numRegions = currentPartition.numRegions;
          depthBeingProcessed = currentPartition.depth;

          thrust::device_ptr<double> wrapped_ptr;
          wrapped_ptr = thrust::device_pointer_cast(dParentsIntegral);
          double parentsEstimate =
            thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions / 2);
          wrapped_ptr = thrust::device_pointer_cast(dParentsError);
          double parentsErrorest =
            thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions / 2);
        }
      }
      else{
         nregions += numRegions + partitionManager.GetNRegions();  
      }

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
      // idea, allocate maximum dParentsIntegral, will it improve performance?
      AllocVolArrays(vol);
      CudaCheckError();
      PrintOutfileHeaders();
      int lastIteration = 0;
      int iteration = 0;
      for (iteration = 0; iteration < 300; iteration++) {
        // printf("----------------------\n");
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

        if ((numRegions < 1 || fail == 0) && iteration != 0) {
          fail = 2;
          break;
        }
        // else if (numRegions > first_phase_maxregions && fail == 1) {
        // else if (numRegions > first_phase_maxregions*8 && fail == 1) {
        else if (iteration == 299 && fail == 1) {
          int last_iteration = 1;
          QuadDebug(cudaFree(dRegionsError));
          QuadDebug(cudaFree(dRegionsIntegral));
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
                                      iteration + 1,
                                      vol,
                                      last_iteration);
          //printf("Hit last iteration\n");
          break;
        } else {
          QuadDebug(cudaFree(dRegionsError));
          QuadDebug(cudaFree(dRegionsIntegral));
        }
        // if(iteration == 24)
        //   PrintToFile(phase1out.str(), "Phase_1_regions.csv");
      }

      // Host.ReleaseMemory(curr_hRegions);
      // Host.ReleaseMemory(curr_hRegionsLength);

      // printf("Host will contain %lu regions\n", numRegions);
      if(outLevel >= 2){
          curr_hRegions =
            (T*)Host.AllocateMemory(&curr_hRegions, sizeof(T) * numRegions * NDIM);
          curr_hRegionsLength = (T*)Host.AllocateMemory(
            &curr_hRegionsLength, sizeof(T) * numRegions * NDIM);

          QuadDebug(cudaMemcpy(curr_hRegions,
                               dRegions,
                               sizeof(T) * numRegions * NDIM,
                               cudaMemcpyDeviceToHost));
          QuadDebug(cudaMemcpy(curr_hRegionsLength,
                               dRegionsLength,
                               sizeof(T) * numRegions * NDIM,
                               cudaMemcpyDeviceToHost));
      }
      
      CudaCheckError();

      /*if(outLevel >= 1){
            PrintToFile(phase1out.str(), "Phase_1_regions.csv");
            PrintToFile(finishedOutfile.str(), "Per_iteration.csv");
      }*/
      StringstreamToFile(finishedOutfile.str(), phase1out.str(), outLevel);

      //---------------
      integral = lastAvg;
      error = lastErr;
      QuadDebug(cudaFree(dRegionsError));
      QuadDebug(cudaFree(dRegionsIntegral));
      //---------------

      bool convergence = false;
      convergence = lastErr <= MaxErr(lastAvg, epsrel, epsabs);
      return !convergence;

      if (fail == 0 || fail == 2) {
        integral = lastAvg;
        error = lastErr;
        QuadDebug(cudaFree(dRegionsError));
        QuadDebug(cudaFree(dRegionsIntegral));
        return true;
      } else {
        return false;
      }
    }

    void
    OutputPhase2Regions(Region<NDIM>* cgRegionPool,
                        T* Regions,
                        T* RegionsLength,
                        int* numRegions,
                        size_t start_size,
                        size_t size)
    {

      std::stringstream outfile;
      std::string filename = "phase2.csv";

      outfile << "value, error, ";
      for (int dim = 0; dim < NDIM; ++dim)
        outfile << "dim" + std::to_string(dim) << ",";
      outfile << "div" << std::endl;

      // outer loop iterates phase 2 block array segments
      for (size_t i = 0; i < start_size; ++i) {

        size_t startIndex = i * max_globalpool_size;
        size_t endIndex = startIndex + numRegions[i];

        for (size_t j = startIndex; j < endIndex; ++j) {

          outfile << cgRegionPool[j].result.avg << ","
                  << cgRegionPool[j].result.err << ",";

          for (int dim = 0; dim < NDIM; ++dim) {
            T lower = Regions[dim * start_size + i];
            T upper = lower + RegionsLength[dim * start_size + i];

            T scaledL =
              lower + cgRegionPool[j].bounds[dim].lower * (upper - lower);
            T scaledU =
              lower + cgRegionPool[j].bounds[dim].upper * (upper - lower);
            outfile << scaledU - scaledL << ",";
            /*if ((scaledU - scaledL) <= 0. || (scaledU - scaledL) >= 1.)
              printf("block:%lu, id:%lu scaled bounds:(%e %e) unscaled:(%e, "
          "%e) div:%i nregions:%i diff:%e\n",
          i,
          j,
          scaledL,
          scaledU,
          cgRegionPool[j].bounds[dim].lower,
          cgRegionPool[j].bounds[dim].upper,
          cgRegionPool[j].div,
          numRegions[i],
          scaledU - scaledL);*/
          }
          outfile << cgRegionPool[j].div << std::endl;
        }
        if (i % 1000 == 0)
          printf("%lu\n", i);
      }

      PrintToFile(outfile.str(), filename);
    }

    void
    OutputPhase2Regions(Region<NDIM>* cgRegionPool,
                        Region<NDIM>* hgRegionsPhase1,
                        int* numRegions,
                        size_t start_size,
                        size_t size)
    {

      std::stringstream outfile;
      std::string filename = "phase2.csv";

      outfile << "value, error, ";
      for (int dim = 0; dim < NDIM; ++dim)
        outfile << "dim" + std::to_string(dim) << ",";
      outfile << "div" << std::endl;

      // outer loop iterates phase 2 block array segments
      double total_vol = 0;
      // double unscaled_total_vol = 0;
      for (size_t i = 0; i < start_size; ++i) {

        size_t startIndex = i * max_globalpool_size;
        size_t endIndex = startIndex + numRegions[i];

        double block_vol = 0;
        // double block_unscaled_vol = 0;

        for (size_t j = startIndex; j < endIndex; ++j) {

          double vol = 1;
          // double unscaled_vol = 1;

          outfile << cgRegionPool[j].result.avg << ","
                  << cgRegionPool[j].result.err << ",";

          for (int dim = 0; dim < NDIM; ++dim) {
            T lower = hgRegionsPhase1[i].bounds[dim].lower;
            T upper = hgRegionsPhase1[i].bounds[dim].upper;

            // unscaled_vol *= upper - lower;

            T scaledL =
              lower + cgRegionPool[j].bounds[dim].lower * (upper - lower);
            T scaledU =
              lower + cgRegionPool[j].bounds[dim].upper * (upper - lower);

            vol *= scaledU - scaledL;

            outfile << scaledU - scaledL << ",";
            /*if ((scaledU - scaledL) <= 0 || (scaledU - scaledL) >= 1 ||
          scaledU - scaledL == 0)
              printf(
          "block:%lu, id:%lu dim:%i scaled bounds:(%f %f) unscaled:(%f, "
          "%f) global bounds:(%f,%f) div:%i nregions:%i diff:%e\n",
          i,
          j,
          dim,
          scaledL,
          scaledU,
          cgRegionPool[j].bounds[dim].lower,
          cgRegionPool[j].bounds[dim].upper,
          lower,
          upper,
          cgRegionPool[j].div,
          numRegions[i],
          scaledU - scaledL);*/
          }

          block_vol += vol;
          // block_unscaled_vol += unscaled_vol;
          outfile << cgRegionPool[j].div + hgRegionsPhase1[i].div << std::endl;
        }
        total_vol += block_vol;
        // unscaled_total_vol += block_unscaled_vol;
      }

      printf("Phase 2 scaled volume:%f\n", total_vol);
      printf("Starting to actually print\n");
      PrintToFile(outfile.str(), filename);
      printf("Finished printing\n");
    }

    template <typename IntegT>
    bool
    IntegrateFirstPhaseDCUHRE(IntegT* d_integrand,
                              T epsrel,
                              T epsabs,
                              T& integral,
                              T& error,
                              size_t& nregions,
                              size_t& neval,
                              Volume<T, NDIM>* vol = nullptr)
    {
      //==============================================================
      // PHASE 1 SETUP
      AllocVolArrays(vol);
      size_t numBlocks = 1;
      size_t numThreads = BLOCK_SIZE;
      // size_t numThreads          = 1;
      size_t numCurrentRegions = 0;

      T* phase1_value = nullptr;
      T* phase1_err = nullptr;

      int* regions_generated = 0;
      int* converged = 0;
      int gpu_id = 0;

      double* dPh1res = nullptr;

      int snapshots[5] = {3968, 4021, 4022, 4023, 4096};
      Snapshot<NDIM> snap(snapshots, 5);
      /*int num = 5;
    QuadDebug(cudaMemcpy(&snap.num,
    &num,
    sizeof(int),
    cudaMemcpyHostToDevice));*/
      CudaCheckError();
      QuadDebug(Device.AllocateMemory((void**)&dPh1res, sizeof(double) * 2));

      cudaMallocManaged((void**)&converged, sizeof(int));
      cudaMallocManaged((void**)&regions_generated, sizeof(int));
      CudaCheckError();
      //==============================================================
      int max_regions = 32734; // worked great for 4e-5 finished in 730 ms
                               // int max_regions  = 16367;
      RegionList* currentBatch = new RegionList;
      cudaMallocManaged(&currentBatch, sizeof(RegionList));
      currentBatch->Set(NDIM,
                        numCurrentRegions,
                        lows,
                        highs,
                        dPh1res,
                        dPh1res + 1,
                        nullptr,
                        converged);
      BLOCK_INTEGRATE_GPU_PHASE2<IntegT, T, NDIM>
        <<<numBlocks, numThreads, NDIM * sizeof(GlobalBounds)>>>(
          d_integrand,
          regions_generated,
          epsrel,
          epsabs,
          gpu_id,
          constMem,
          rule.GET_FEVAL(),
          rule.GET_NSETS(),
          lows,
          highs,
          Final,
          gRegionPool,
          nullptr,
          nullptr,
          0,
          0,
          0,
          0,
          dPh1res,
          max_regions,
          *currentBatch,
          phase_I_type,
          0,
          snap);

      PrintOutfileHeaders();
      numRegions = max_regions;
      CudaCheckError();
      cudaDeviceSynchronize();

      // snap.Save("pdc256_snapshot");
      /*Region<NDIM>* tempGregions = (Region<NDIM>*)malloc(sizeof(Region<NDIM>)
       * max_regions);


       QuadDebug(cudaMemcpy(tempGregions,
       gRegionPool,
       sizeof(Region<NDIM>) * max_regions,
       cudaMemcpyDeviceToHost));

       std::ofstream outfile("dc_phase1.csv");
       outfile.precision(20);
       for(int i=0; i< max_regions; i++){
       outfile<<tempGregions[i].result.avg<<","<<tempGregions[i].result.err<<",";
       for(int dim = 0; dim < NDIM; dim++){
       outfile<<tempGregions[i].bounds[dim].upper -
       tempGregions[i].bounds[dim].lower<<",";
       }
       outfile<<tempGregions[i].result.bisectdim<<","<<tempGregions[i].div<<std::endl;
       }
       outfile.close();*/

      // if (outLevel >= 1)

      // T hphase1_value         = 0;//(T*)malloc(sizeof(T));
      // T hphase1_err       = 0;//(T*)malloc(sizeof(T));
      int curr_hRegions_generated = 0; //(int*)malloc(sizeof(int));

      QuadDebug(
        cudaMemcpy(&lastAvg, dPh1res, sizeof(T), cudaMemcpyDeviceToHost));

      QuadDebug(
        cudaMemcpy(&lastErr, dPh1res + 1, sizeof(T), cudaMemcpyDeviceToHost));

      QuadDebug(cudaMemcpy(&curr_hRegions_generated,
                           regions_generated,
                           sizeof(int),
                           cudaMemcpyDeviceToHost));

      if (outLevel == 4) {
        Region<NDIM>* tmp = 0;
        std::ofstream outfile("phase1.csv");
        tmp = (Region<NDIM>*)malloc(sizeof(Region<NDIM>) * max_regions);
        cudaMemcpy(tmp,
                   gRegionPool,
                   sizeof(Region<NDIM>) * max_regions,
                   cudaMemcpyDeviceToHost);
        double tot_vol = 0;
        outfile << "value, error, dim0,dim1,dim2,dim3,dim4,div" << std::endl;

        for (int i = 0; i < max_regions; ++i) {
          double vol = 1;
          outfile << tmp[i].result.avg << "," << tmp[i].result.err << ",";
          for (int dim = 0; dim < NDIM; dim++) {
            vol *= tmp[i].bounds[dim].upper - tmp[i].bounds[dim].lower;
            outfile << tmp[i].bounds[dim].upper - tmp[i].bounds[dim].lower
                    << ",";
          }
          outfile << tmp[i].div << std::endl;
          tot_vol += vol;
        }
        outfile.close();
        free(tmp);
      }

      if (outLevel >= 1)
        out1 << lastAvg << "," << lastErr << "," << curr_hRegions_generated
             << std::endl;

      CudaCheckError();
      // std::cout <<"Phase 1 Result:"<< lastAvg << ", " << lastErr << ", " <<
      // curr_hRegions_generated << "ratio:" << lastErr/MaxErr(lastAvg, epsrel,
      // epsabs) << std::endl;
      // printf("Phase 1 Result:%.20f +- %.20f regions:%i ratio:%f\n", lastAvg,
      // lastErr, curr_hRegions_generated, lastErr/MaxErr(lastAvg, epsrel,
      // epsabs));
      nregions = curr_hRegions_generated;

      cudaFree(phase1_value);
      cudaFree(phase1_err);
      cudaFree(converged);
      cudaFree(regions_generated);

      CudaCheckError();

      // printf("Phase 1 %.15f +- %.15f ratio:%f\n", lastAvg,
      // lastErr,lastErr/MaxErr(lastAvg, epsrel, epsabs));
      if (lastErr / MaxErr(lastAvg, epsrel, epsabs) < 1) {
        fail = 0;
        integral = lastAvg;
        error = lastErr;
        return true;
      }

      CudaCheckError();
      CudaCheckError();
      QuadDebug(Device.ReleaseMemory(dPh1res));
      QuadDebug(
        Device.AllocateMemory((void**)&dRegionsError, sizeof(T) * numRegions));
      CudaCheckError();
      QuadDebug(Device.AllocateMemory((void**)&dRegionsIntegral,
                                      sizeof(T) * numRegions));
      CudaCheckError();
      return false;
    }

    PhaseII_output Evaluate(RegionList* batch, int* dRegionsNumRegion/*, double& integral, double& error, int& regionCnt, int& numFailedRegions*/)
    {
      PhaseII_output result;
      result.num_starting_blocks = batch->numRegions;
      thrust::device_ptr<T> wrapped_ptr;

      wrapped_ptr = thrust::device_pointer_cast(batch->dRegionsIntegral);
      result.estimate =
        thrust::reduce(wrapped_ptr, wrapped_ptr + batch->numRegions);

      wrapped_ptr = thrust::device_pointer_cast(batch->dRegionsError);
      result.errorest =
        thrust::reduce(wrapped_ptr, wrapped_ptr + batch->numRegions);

      thrust::device_ptr<int> int_ptr =
        thrust::device_pointer_cast(dRegionsNumRegion);
      result.regions = thrust::reduce(int_ptr, int_ptr + batch->numRegions);

      int_ptr = thrust::device_pointer_cast(batch->activeRegions);
      result.num_failed_blocks =
        thrust::reduce(int_ptr, int_ptr + batch->numRegions);

      printf("Batch Results:%f +- %f nregions:%i, numFailedRegions:%i\n",
             result.estimate,
             result.errorest,
             result.regions,
             result.num_failed_blocks);
      return result;
    }

    template <typename IntegT>
    PhaseII_output
    Execute_PhaseII_Batch(RegionList*& batch,
                          int* dRegionsNumRegion,
                          int gpu_id,
                          double* Phase_I_result,
                          Region<NDIM>* ph1_regions,
                          cudaStream_t stream,
                          double epsrel,
                          double epsabs,
                          IntegT* d_integrand)
    {

      int max_regions = max_globalpool_size;
      size_t numThreads = BLOCK_SIZE;
      size_t numBlocks = batch->numRegions;
      // printf("Lauching %lu phase 2 blocks\n", numBlocks);
      BLOCK_INTEGRATE_GPU_PHASE2<IntegT, T, NDIM>
        <<<numBlocks, numThreads, NDIM * sizeof(GlobalBounds), stream>>>(
          d_integrand,
          dRegionsNumRegion,
          epsrel,
          epsabs,
          gpu_id,
          constMem,
          rule.GET_FEVAL(),
          rule.GET_NSETS(),
          lows,
          highs,
          Final,
          gRegionPool,
          nullptr,
          nullptr,
          lastAvg,
          lastErr,
          weightsum,
          avgsum,
          Phase_I_result,
          max_regions,
          *batch,
          phase_I_type,
          ph1_regions, // used for alternative phase 1
          Snapshot<NDIM>(),
          nullptr,
          nullptr);

      cudaDeviceSynchronize();
      CudaCheckError();

      // DONT FORGET FINAL=0 ADJUSTMENT ON RESULT

      // double batch_estimate = 0., batch_errorest = 0.;
      // int batch_regions = 0.;
      // int num_failed_blocks = 0;
      return Evaluate(batch, dRegionsNumRegion/*, batch_estimate, batch_errorest, batch_regions, num_failed_blocks*/);
    }

    void
    Phase_I_format_region_copy(
      RegionList*& batch,
      double*& sourceRegions,       // should not be by reference, const instead
      double*& sourceRegionsLength, // should not be by reference, const instead
      double* RegionsIntegral,      // should not be by reference, const instead
      double* RegionsError,         // should not be by reference, const instead
      size_t total_num_regions,
      size_t startRegionIndex,
      size_t endRegionIndex)
    {

      // from a two based region array, copy a partion into batch
      // size_t batch_size = endRegionIndex - startRegionIndex + 1;
      size_t batch_size = endRegionIndex - startRegionIndex + 1;
      CudaCheckError();
      assert(batch_size < 0);

      batch->numRegions = batch_size;
      batch->dRegionsIntegral = RegionsIntegral;
      batch->dRegionsError = RegionsError;
      CudaCheckError();

      for (int dim = 0; dim < NDIM; ++dim) {
        // printf("Copying regions: from %lu to %lu\n", dim * total_num_regions
        // + startRegionIndex, dim * total_num_regions +
        // startRegionIndex+batch_size);
        QuadDebug(
          cudaMemcpy(batch->dRegions + dim * batch_size,
                     sourceRegions + dim * total_num_regions + startRegionIndex,
                     sizeof(T) * batch_size,
                     cudaMemcpyDeviceToDevice));

        QuadDebug(cudaMemcpy(
          batch->dRegionsLength + dim * batch_size,
          sourceRegionsLength + dim * total_num_regions + startRegionIndex,
          sizeof(T) * batch_size,
          cudaMemcpyDeviceToDevice)); // this is not HostToDevice right?
        CudaCheckError();
      }
    }

    template <typename IntegT>
    PhaseII_output
    Execute_PhaseII_Batches(double* Regions,
                            double* RegionsLength,
                            size_t size,
                            int gpu_id,
                            Region<NDIM>* ph1_regions,
                            cudaStream_t stream,
                            IntegT* d_integrand,
                            double epsrel,
                            double epsabs,
                            int* dRegionsNumRegion,
                            double* Phase_I_result)
    {

      PhaseII_output final_result;

      size_t max_num_blocks = FIRST_PHASE_MAXREGIONS * 2;
      RegionList* currentBatch =
        new RegionList(NDIM, size); // why have the entire
      currentBatch->Set(
        dRegionsIntegral,
        dRegionsError); // at first we point to the entire thing, shallow copy
      size_t start = 0;
      // size_t end = max_num_blocks;
      int iters = size / max_num_blocks;
      printf("Will require %i phase II iterations\n", iters);

      for (int it = 0; it < iters; it++) {
        size_t leftIndex = start + it * max_num_blocks;
        size_t rightIndex = leftIndex + max_num_blocks - 1;

        // printf("Assigning dRegionsIntegral[%lu] to batch %i\n", numRegions +
        // leftIndex, it);
        Phase_I_format_region_copy(
          currentBatch,
          Regions,
          RegionsLength,
          dRegionsIntegral + numRegions + leftIndex,
          dRegionsError + numRegions + leftIndex,
          size,
          leftIndex,
          rightIndex); // extract the coordinate info, deep copy
        CudaCheckError();
        final_result += Execute_PhaseII_Batch(currentBatch,
                                              dRegionsNumRegion,
                                              gpu_id,
                                              Phase_I_result,
                                              ph1_regions,
                                              stream,
                                              epsrel,
                                              epsabs,
                                              d_integrand);
        CudaCheckError();
        currentBatch->Clear();
        CudaCheckError();
      }

      if (size % max_num_blocks != 0) {

        size_t leftIndex = start + iters * max_num_blocks;
        size_t rightIndex = leftIndex + (size % max_num_blocks) - 1;
        Phase_I_format_region_copy(currentBatch,
                                   Regions,
                                   RegionsLength,
                                   dRegionsIntegral + numRegions + leftIndex,
                                   dRegionsError + numRegions + leftIndex,
                                   size,
                                   leftIndex,
                                   rightIndex);
        final_result += Execute_PhaseII_Batch(currentBatch,
                                              dRegionsNumRegion,
                                              gpu_id,
                                              Phase_I_result,
                                              ph1_regions,
                                              stream,
                                              epsrel,
                                              epsabs,
                                              d_integrand);
        CudaCheckError();
        currentBatch->Clear();
        CudaCheckError();
      }

      return final_result;
    }

    // This is supposed to work on all regions, ? maybe have a regionList for
    // that too? why not
    void
    Assing_Regions_To_Processor(double*& dRegionsThread,
                                double*& dRegionsLengthThread,
                                size_t numRegions,
                                unsigned int cpu_thread_id,
                                unsigned int num_cpu_threads)
    {
      // This is OpenMP aware distribution of regions to one or more processors
      // it assigns all regions that will ever be processed in phase 2 by a
      // processor the batching doesn't happen here
      size_t numRegionsThread = numRegions / num_cpu_threads;
      int startIndex = cpu_thread_id * numRegionsThread;
      int endIndex = (cpu_thread_id + 1) * numRegionsThread;

      if (cpu_thread_id == (num_cpu_threads - 1))
        endIndex = numRegions;

      numRegionsThread = endIndex - startIndex;

      for (int dim = 0; dim < NDIM; ++dim) {
        QuadDebug(cudaMemcpy(dRegionsThread + dim * numRegionsThread,
                             curr_hRegions + dim * numRegions + startIndex,
                             sizeof(T) * numRegionsThread,
                             cudaMemcpyHostToDevice));

        QuadDebug(
          cudaMemcpy(dRegionsLengthThread + dim * numRegionsThread,
                     curr_hRegionsLength + dim * numRegions + startIndex,
                     sizeof(T) * numRegionsThread,
                     cudaMemcpyHostToDevice));
      }
    }

    template <typename IntegT>
    size_t
    IntegrateSecondPhase(IntegT* d_integrand,
                         T epsrel,
                         T epsabs,
                         T& integral,
                         T& error,
                         size_t& nregions,
                         size_t& neval /*,T* optionalInfo = 0*/)
    {
      int num_gpus = 0; // number of CUDA GPUs
      int numFailedRegions = 0;

      cudaGetDeviceCount(&num_gpus);
      if (num_gpus < 1) {
        fprintf(stderr, "no CUDA capable devices were detected\n");
        exit(1);
      }

      int num_cpu_procs = omp_get_num_procs();

      if (NUM_DEVICES > num_gpus)
        NUM_DEVICES = num_gpus;

      omp_set_num_threads(NUM_DEVICES);
      cudaStream_t stream[NUM_DEVICES];
      cudaEvent_t event[NUM_DEVICES];
      CudaCheckError();
#pragma omp parallel

      {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;

        // "% num_gpus" allows more CPU threads than GPU devices
        QuadDebug(cudaSetDevice(cpu_thread_id % num_gpus));
        QuadDebug(cudaGetDevice(&gpu_id));
        warmUpKernel<<<first_phase_maxregions, BLOCK_SIZE>>>();

        if (cpu_thread_id < num_cpu_threads) {
          // std::cout<<"Phase 2 has "<<numRegions<<" regions and
          // "<<num_cpu_threads<< " cpu threads\n";
          size_t numRegionsThread = numRegions / num_cpu_threads;
          int startIndex = cpu_thread_id * numRegionsThread;
          int endIndex = (cpu_thread_id + 1) * numRegionsThread;

          if (cpu_thread_id == (num_cpu_threads - 1))
            endIndex = numRegions;

          numRegionsThread = endIndex - startIndex;

          CudaCheckError();

          T *dRegionsThread = 0, *dRegionsLengthThread = 0;
          int* dRegionsNumRegion = 0;

          QuadDebug(Device.AllocateUnifiedMemory(
            (void**)&dRegionsNumRegion, sizeof(int) * numRegionsThread));

          if (phase_I_type == 0) {
            // dRegionsThread & dRegionsLengthThread store all regions and get
            // contents (partially or fully) from curr_hRegions and
            // curr_hRegionsLength
            QuadDebug(Device.AllocateUnifiedMemory(
              (void**)&dRegionsThread, sizeof(T) * numRegionsThread * NDIM));
            QuadDebug(Device.AllocateUnifiedMemory(
              (void**)&dRegionsLengthThread,
              sizeof(T) * numRegionsThread * NDIM));

            // this function creates a batch out of all regins, ready to be
            // passed as an object to Phase 2 Kernel
            Assing_Regions_To_Processor(dRegionsThread,
                                        dRegionsLengthThread,
                                        numRegions,
                                        cpu_thread_id,
                                        num_cpu_threads);
            CudaCheckError();
          }

          // RegionList* currentBatch = new RegionList(NDIM, numRegionsThread);
          // currentBatch->Set(NDIM, numRegionsThread, dRegionsThread,
          // dRegionsLengthThread, dRegionsIntegral, dRegionsError);
          CudaCheckError();

          cudaEvent_t start;
          QuadDebug(cudaStreamCreate(&stream[gpu_id]));
          QuadDebug(cudaEventCreate(&start));
          QuadDebug(cudaEventCreate(&event[gpu_id]));
          QuadDebug(cudaEventRecord(start, stream[gpu_id]));
          CudaCheckError();

          // this is where batch execution starts
          double* Phase_I_result = nullptr;
          QuadDebug(
            Device.AllocateMemory((void**)&Phase_I_result, sizeof(double) * 2));
          QuadDebug(cudaMemcpy(
            Phase_I_result, &lastAvg, sizeof(T), cudaMemcpyHostToDevice));
          QuadDebug(cudaMemcpy(
            Phase_I_result + 1, &lastErr, sizeof(T), cudaMemcpyHostToDevice));

          int start_regions = 32734;
          Region<NDIM>* ph1_regions = 0;
          if (phase_I_type == 1) {
            QuadDebug(Device.AllocateMemory(
              (void**)&ph1_regions, sizeof(Region<NDIM>) * start_regions));
            cudaMemcpy(ph1_regions,
                       gRegionPool,
                       sizeof(Region<NDIM>) * start_regions,
                       cudaMemcpyDeviceToDevice);
          }

          CudaCheckError();

          Region<NDIM>* tmp = nullptr;
          if (outLevel >= 4 && phase_I_type == 1) {
            printf("Entered outlevel >=4\n");
            std::stringstream tempOut;
            tmp = (Region<NDIM>*)malloc(sizeof(Region<NDIM>) * start_regions);
            cudaMemcpy(tmp,
                       ph1_regions,
                       sizeof(Region<NDIM>) * start_regions,
                       cudaMemcpyDeviceToHost);
            double tot_vol = 0;
            tempOut << "value, error, dim0,dim1,dim2,dim3,dim4,div"
                    << std::endl;

            for (int i = 0; i < start_regions; ++i) {
              double vol = 1;
              tempOut << tmp[i].result.avg << "," << tmp[i].result.err << ",";
              if (i == 0)
                std::cout << tmp[i].result.avg << "," << tmp[i].result.err
                          << ",";
              for (int dim = 0; dim < NDIM; dim++) {
                vol *= tmp[i].bounds[dim].upper - tmp[i].bounds[dim].lower;
                tempOut << tmp[i].bounds[dim].upper - tmp[i].bounds[dim].lower
                        << ",";
                if (i == 0)
                  std::cout
                    << tmp[i].bounds[dim].upper - tmp[i].bounds[dim].lower
                    << ",";
              }
              tempOut << tmp[i].div << std::endl;
              if (i == 0)
                std::cout << tmp[i].div << std::endl;
              tot_vol += vol;
            }
            PrintToFile(tempOut.str(), "phase1.csv");
          }

          printf("Phase 1 temp results: %.17f +- %.17f ratio:%f nregions:%lu\n",
                 lastAvg,
                 lastErr,
                 lastErr / MaxErr(lastAvg, epsrel, epsabs),
                 numRegions);
          printf("Phase 1 good region results:%.17f +- %.17f nregions:%lu\n",
                 integral,
                 error,
                 numInActiveRegions);
          printf("-------\n");

          // int max_regions = max_globalpool_size;
          // size_t numThreads = BLOCK_SIZE;
          size_t numBlocks = numRegionsThread;

          CudaCheckError();

          // store good region phase 1 results in order to increment them with
          // phase 2 results
          PhaseII_output phase_II_final_output;
          phase_II_final_output.estimate = 0.;
          phase_II_final_output.errorest = 0.;
          phase_II_final_output.regions = 0;
          phase_II_final_output.num_failed_blocks = 0;
          phase_II_final_output.num_starting_blocks = 0;
          // CALL EXECUTE BATCH HERE
          phase_II_final_output += Execute_PhaseII_Batches(dRegionsThread,
                                                           dRegionsLengthThread,
                                                           numBlocks,
                                                           gpu_id,
                                                           ph1_regions,
                                                           stream[gpu_id],
                                                           d_integrand,
                                                           epsrel,
                                                           epsabs,
                                                           dRegionsNumRegion,
                                                           Phase_I_result);

          CudaCheckError();

          cudaEventRecord(event[gpu_id], stream[gpu_id]);
          cudaEventSynchronize(event[gpu_id]);

          float elapsed_time;
          cudaEventElapsedTime(&elapsed_time, start, event[gpu_id]);

          cudaEventDestroy(start);
          cudaEventDestroy(event[gpu_id]);

          integral += phase_II_final_output.estimate,
            error += phase_II_final_output.errorest;
          nregions += phase_II_final_output.regions;
          CudaCheckError();

          /*if (Final == 0) {
            double w = numRegionsThread * 1 / fmax(error * error, ldexp(1.,
        -104)); weightsum += w; // adapted by Ioannis avgsum += w * integral;
            double sigsq = 1 / weightsum;
            integral = sigsq * avgsum;
            error = sqrt(sigsq);
        }*/

          if (phase_I_type == 0) {
            CudaCheckError();
            QuadDebug(Device.ReleaseMemory(dRegionsThread));
            CudaCheckError();
            QuadDebug(Device.ReleaseMemory(dRegionsLengthThread));
            CudaCheckError();
          }
          QuadDebug(Device.ReleaseMemory(Phase_I_result));
          CudaCheckError();
          QuadDebug(Device.ReleaseMemory(dRegionsNumRegion));
          CudaCheckError();
        }

        else
          printf("Rogue cpu thread\n");
      }

      // free conditional allocations

      return numFailedRegions;
    }
  };

}
#endif
