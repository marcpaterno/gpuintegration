#ifndef CUDACUHRE_QUAD_GPUQUAD_KERNEL_CUH
#define CUDACUHRE_QUAD_GPUQUAD_KERNEL_CUH

#include "cuda/pagani/quad/util/Volume.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>

#include "cuda/pagani/quad/GPUquad/Phases.cuh"
#include "cuda/pagani/quad/GPUquad/Rule.cuh"
#include "cuda/pagani/quad/GPUquad/Func_Eval.cuh"

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
#include "cuda/pagani/quad/util/mem_util.cuh"


namespace quad {

  //===========
  // FOR DEBUGGINGG
  void
  print_to_file(std::string outString,
                std::string filename,
                bool appendMode = 0)
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

  template <typename T>
  __global__ void
  generateInitialRegions(T* dRegions,
                         T* dRegionsLength,
                         size_t numRegions,
                         T* newRegions,
                         T* newRegionsLength,
                         size_t newNumOfRegions,
                         int numOfDivisionsPerRegionPerDimension,
                         int ndim)
  {
    extern __shared__ T slength[];
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < ndim) {
      slength[threadIdx.x] =
        dRegionsLength[threadIdx.x] / numOfDivisionsPerRegionPerDimension;
    }
    __syncthreads();

    if (threadId >= newNumOfRegions)
      return;

    size_t interval_index =
      threadId / pow((T)numOfDivisionsPerRegionPerDimension, (T)ndim);
    size_t local_id =
      threadId % (size_t)pow((T)numOfDivisionsPerRegionPerDimension, (T)ndim);
    for (int dim = 0; dim < ndim; ++dim) {
      size_t id =
        (size_t)(local_id /
                 pow((T)numOfDivisionsPerRegionPerDimension, (T)dim)) %
        numOfDivisionsPerRegionPerDimension;

      newRegions[newNumOfRegions * dim + threadId] =
        dRegions[numRegions * dim + interval_index] + id * slength[dim];
      newRegionsLength[newNumOfRegions * dim + threadId] = slength[dim];
    }
  }

  template <typename T>
  __global__ void
  alignRegions(int ndim,
               T* dRegions,
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

      for (int i = 0; i < ndim; ++i) {
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

  template <typename T>
  __global__ void
  divideIntervalsGPU(int ndim,
                     T* genRegions,
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
        for (int dim = 0; dim < ndim; ++dim) {
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

  template <typename T, int NDIM>
  class Kernel {
    // these are also members of RegionList
    T* dRegionsError;
    T* dRegionsIntegral;
    T* dRegions;
    T* dRegionsLength;

    T* curr_hRegions;
    T* curr_hRegionsLength;

    T* dParentsError;
    T* dParentsIntegral;

    //-----------------------------------

    T* highs;
    T* lows;

    Region<NDIM>* gRegionPool;
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
    int heuristicID;
    int fail; // 0 for satisfying ratio, 1 for not satisfying ratio, 2 for
              // running out of bad regions
    T lastErr;
    T lastAvg;
    T secondTolastAvg;

    T estimate_change;

    int key, verbose, outLevel;
    size_t numRegions, h_numRegions, numInActiveRegions;
    size_t fEvalPerRegion;
    int first_phase_maxregions;
    int max_globalpool_size;
    bool mustFinish;
    size_t nextAvailRegionID; // make it a local variable to kernel fucntion, no
                              // need to bloat the class
    size_t nextAvailParentID; // same goes here
    size_t* parentIDs;        // same goes here

    HostMemory<T> Host;
    DeviceMemory<T> Device;
    Rule<T> rule;
    Structures<T> constMem;
    int NUM_DEVICES;
    // Debug Msg
    char msg[256];

    T* generators = nullptr;

  public:
    void
    GetPtrsToArrays(T*& regions,
                    T*& regionsLength,
                    T*& regionsIntegral,
                    T*& regionsError,
                    T*& gener)
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
            Structures<T>*& constMemory,
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
    GetVolumeBounds(T* lowBounds, T* highBounds)
    {
      lowBounds = lows;
      highBounds = highs;
    }

    T
    GetIntegral()
    {
      return lastAvg;
    }

    T
    GetError()
    {
      return lastErr;
    }

    int
    GetErrorFlag()
    {
      return fail;
    }

    T
    GetRatio(T epsrel, T epsabs)
    {
      return lastErr / MaxErr(lastAvg, epsrel, epsabs);
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
    ExpandcuArray(T*& array, int currentSize, int newSize)
    {
      T* temp = 0;
      int copy_size = std::min(currentSize, newSize);
      // printf("current size:%i, newSize:%i\n", currentSize, newSize);
      QuadDebug(Device.AllocateMemory((void**)&temp, sizeof(T) * newSize));
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
    }

    Kernel()
    {
      mustFinish = false;
      dParentsError = nullptr;
      dParentsIntegral = nullptr;
      gRegionPool = nullptr;
      estimate_change = 0.;
      estimateHasConverged = false;

      ConfigureMemoryUtilization();

      lastErr = 0;
      lastAvg = 0;
      fail = 1;
      numRegions = 0;
      key = 0;
      h_numRegions = 0;
    }

    ~Kernel()
    {
      CudaCheckError();
      // dRegions and dRegionsLength need to be freed after phase 1, since all
      // the info is stored in host memory

      QuadDebug(cudaFree(constMem.gpuG));
      QuadDebug(cudaFree(constMem.cRuleWt));
      QuadDebug(cudaFree(constMem.GPUScale));
      QuadDebug(cudaFree(constMem.GPUNorm));
      QuadDebug(cudaFree(constMem.gpuGenPos));
      QuadDebug(cudaFree(constMem.gpuGenPermGIndex));
      QuadDebug(cudaFree(constMem.gpuGenPermVarStart));
      QuadDebug(cudaFree(constMem.gpuGenPermVarCount));
      QuadDebug(cudaFree(constMem.cGeneratorCount));
      CudaCheckError();
    }

    void
    StringstreamToFile(std::string per_iteration,
                       std::string per_region,
                       int verbosity)
    {
      switch (verbosity) {
        case 1:
          print_to_file(per_iteration,
                        "h" + std::to_string(heuristicID) +
                          "_Per_iteration.csv");
          break;
        case 2:
          printf("Printing for verbosity 2\n");
          print_to_file(per_iteration,
                        "h" + std::to_string(heuristicID) +
                          "_Per_iteration.csv");
          print_to_file(per_region, "Phase_1_regions.csv");
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
    InitKernel(int key, int new_verbosity)
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
      key = key;
      verbose = new_verbosity;
      fEvalPerRegion = (1 + 2 * NDIM + 2 * NDIM + 2 * NDIM + 2 * NDIM +
                        2 * NDIM * (NDIM - 1) + 4 * NDIM * (NDIM - 1) +
                        4 * NDIM * (NDIM - 1) * (NDIM - 2) / 3 + (1 << NDIM));
      /*QuadDebug(cudaMemcpyToSymbol(dFEvalPerRegion,
                                   &fEvalPerRegion,
                                   sizeof(size_t),
                                   0,
                                   cudaMemcpyHostToDevice));*/
      rule.Init(NDIM, fEvalPerRegion, key, verbose, &constMem);
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

      if (outLevel >= 1) {
        printf("OutLevel 1\n");
        print_to_file(out1.str(), "Level_1.csv");
      }

      if (outLevel >= 3) {
        printf("OutLevel 3\n");

        auto callback = [](T integral, T error, T rel) {
          return fabs(error / (rel * integral));
        };

        using func_pointer = T (*)(T integral, T error, T rel);
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
          T val = tmp[i].result.avg;
          T err = tmp[i].result.err;
          out3 << val << "," << err << "," << err / MaxErr(val, epsrel, epsabs)
               << std::endl;
        }
        print_to_file(out3.str(), "start_ratio.csv");
        free(tmp);
      }
    }

    void
    PrintIteration(int* activeRegions,
                   int iteration,
                   size_t iter_nregions,
                   T leaves_estimate,
                   T leaves_errorest,
                   T iter_estimate,
                   T iter_errorest,
                   T iter_finished_estimate,
                   T iter_finished_errorest /*,
                    T queued_estimate,
                    T queued_errorest,
                    size_t unevaluated_nregions*/
    )
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
                      << ","
                      << dnumInActiveRegions /*<< "," << queued_estimate*/
                      //<< "," << queued_errorest << "," << unevaluated_nregions
                      << "\n";

      std::cout << std::setprecision(17) << std::scientific << heuristicID
                << "," << iteration << "," << leaves_estimate << ","
                << leaves_errorest << "," << iter_nregions << ","
                << iter_estimate << "," << iter_errorest << ","
                << iter_finished_estimate << "," << iter_finished_errorest
                << "," << dnumInActiveRegions /*<< "," << queued_estimate << ","
                << queued_errorest << "," << unevaluated_nregions*/
                << "\n";
      Device.ReleaseMemory(scannedArray);
    }

    void
    Phase_I_PrintFile(Volume<T, NDIM> const* vol,
                      size_t iter_nregions,
                      int* activeRegions,
                      T leaves_estimate,
                      T leaves_errorest,
                      T iter_estimate,
                      T iter_errorest,
                      T iter_finished_estimate,
                      T iter_finished_errorest,
                      T epsrel,
                      T epsabs,
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
                     iter_finished_errorest);

      if (outLevel < 2)
        return;

      size_t numActiveRegions = 0;
      int* h_activeRegions = nullptr;
      h_activeRegions = (int*)Host.AllocateMemory(
        h_activeRegions,
        sizeof(int) *
          iter_nregions); //(int*)malloc(sizeof(int) * iter_nregions);

      T* curr_hRegionsIntegral = nullptr;
      T* curr_hRegionsError = nullptr;
      T* curr_ParentsIntegral = nullptr;
      T* curr_ParentsError = nullptr;

      T* h_highs = nullptr;
      T* h_lows = nullptr;

      curr_hRegionsIntegral = (T*)Host.AllocateMemory(
        curr_hRegionsIntegral, sizeof(T) * iter_nregions);
      curr_hRegionsError =
        (T*)Host.AllocateMemory(curr_hRegionsError, sizeof(T) * iter_nregions);
      curr_ParentsIntegral = (T*)Host.AllocateMemory(
        curr_ParentsIntegral, sizeof(T) * ceil(iter_nregions / 2));
      curr_ParentsError = (T*)Host.AllocateMemory(
        curr_ParentsError, sizeof(T) * ceil(iter_nregions / 2));
      h_highs = (T*)Host.AllocateMemory(h_highs, sizeof(T) * NDIM);
      h_lows = (T*)Host.AllocateMemory(h_lows, sizeof(T) * NDIM);

      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_hRegionsIntegral,
                           dRegionsIntegral,
                           sizeof(T) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_hRegionsError,
                           dRegionsError,
                           sizeof(T) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();

      QuadDebug(cudaMemcpy(curr_ParentsIntegral,
                           dParentsIntegral,
                           sizeof(T) * ceil(iter_nregions / 2),
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(cudaMemcpy(curr_ParentsError,
                           dParentsError,
                           sizeof(T) * ceil(iter_nregions / 2),
                           cudaMemcpyDeviceToHost));
      CudaCheckError();

      QuadDebug(cudaMemcpy(h_activeRegions,
                           activeRegions,
                           sizeof(int) * iter_nregions,
                           cudaMemcpyDeviceToHost));
      CudaCheckError();
      QuadDebug(
        cudaMemcpy(h_highs, highs, sizeof(T) * NDIM, cudaMemcpyDeviceToHost));

      QuadDebug(
        cudaMemcpy(h_lows, lows, sizeof(T) * NDIM, cudaMemcpyDeviceToHost));

      CudaCheckError();

      curr_hRegions = (T*)Host.AllocateMemory(curr_hRegions,
                                              sizeof(T) * iter_nregions * NDIM);
      curr_hRegionsLength = (T*)Host.AllocateMemory(
        curr_hRegionsLength, sizeof(T) * iter_nregions * NDIM);

      QuadDebug(cudaMemcpy(curr_hRegions,
                           dRegions,
                           sizeof(T) * iter_nregions * NDIM,
                           cudaMemcpyDeviceToHost));
      QuadDebug(cudaMemcpy(curr_hRegionsLength,
                           dRegionsLength,
                           sizeof(T) * iter_nregions * NDIM,
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

        T ratio = curr_hRegionsError[regnIndex] /
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
          T low = ScaleValue(curr_hRegions[dim * iter_nregions + regnIndex],
                             h_lows[dim],
                             h_highs[dim]);

          T high =
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
        sizeof(T) *
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
      // reset variables before allocating host memory
      mustFinish = false;
      dParentsError = nullptr;
      dParentsIntegral = nullptr;
      gRegionPool = nullptr;
      estimate_change = 0.;
      estimateHasConverged = false;

      ConfigureMemoryUtilization();

      lastErr = 0;
      lastAvg = 0;
      fail = 1;
      numRegions = 0;
      key = 0;
      h_numRegions = 0;

      curr_hRegions = (T*)Host.AllocateMemory(&curr_hRegions, sizeof(T) * NDIM);
      curr_hRegionsLength =
        (T*)Host.AllocateMemory(&curr_hRegionsLength, sizeof(T) * NDIM);

      for (int dim = 0; dim < NDIM; ++dim) {
        curr_hRegions[dim] = 0;
        curr_hRegionsLength[dim] = 1;
      }

      dRegions = cuda_malloc<T>(NDIM);
      dRegionsLength = cuda_malloc<T>(NDIM);
      CudaCheckError();
      QuadDebug(cudaMemcpy(
        dRegions, curr_hRegions, sizeof(T) * NDIM, cudaMemcpyHostToDevice));
      QuadDebug(cudaMemcpy(dRegionsLength,
                           curr_hRegionsLength,
                           sizeof(T) * NDIM,
                           cudaMemcpyHostToDevice));
      CudaCheckError();
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
      CudaCheckError();
      size_t numBlocks = (size_t)ceil(
        pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM) / numThreads);
      numRegions = (size_t)pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM);

      T* newRegions = cuda_malloc<T>(numRegions * NDIM);
      T* newRegionsLength = cuda_malloc<T>(numRegions * NDIM);
      CudaCheckError();

      generateInitialRegions<T><<<numBlocks, numThreads, NDIM * sizeof(T)>>>(
        dRegions,
        dRegionsLength,
        1,
        newRegions,
        newRegionsLength,
        numRegions,
        numOfDivisionPerRegionPerDimension,
        NDIM);
      cudaFree(dRegions);
      cudaFree(dRegionsLength);

      dRegions = newRegions;
      dRegionsLength = newRegionsLength;

      Host.ReleaseMemory(curr_hRegions);
      Host.ReleaseMemory(curr_hRegionsLength);
    }

    size_t
    GenerateActiveIntervals(int* activeRegions,
                            int* subDividingDimension,
                            T* dRegionsIntegral,
                            T* dRegionsError,
                            T*& dParentsIntegral,
                            T*& dParentsError)
    {
      //nvtxRangePush("Compute numActive Regions");
      int* scannedArray = 0; // de-allocated at the end of this function
      QuadDebug(
        Device.AllocateMemory((void**)&scannedArray, sizeof(int) * numRegions));

      thrust::device_ptr<int> d_ptr =
        thrust::device_pointer_cast(activeRegions);

      CudaCheckError();

      thrust::device_ptr<int> scan_ptr =
        thrust::device_pointer_cast(scannedArray);
      //nvtxRangePush("Exclusive scan");
      thrust::exclusive_scan(d_ptr, d_ptr + numRegions, scan_ptr);
      //nvtxRangePop();
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

      //nvtxRangePop();

      numInActiveRegions = numRegions - numActiveRegions;

      if (outLevel >= 4)
        out4 << numActiveRegions << "," << numRegions << std::endl;

      if (numActiveRegions > 0) {
        //nvtxRangePush("Aligning Regions");
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

        alignRegions<T><<<numBlocks, numThreads>>>(NDIM,
                                                   dRegions,
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

        CudaCheckError();
        //nvtxRangePop();
        T *genRegions = 0, *genRegionsLength = 0;
        numBlocks = numActiveRegions / numThreads +
                    ((numActiveRegions % numThreads) ? 1 : 0);

        QuadDebug(Device.ReleaseMemory(dRegions));
        QuadDebug(Device.ReleaseMemory(dRegionsLength));
        QuadDebug(Device.ReleaseMemory(scannedArray));

        //nvtxRangePush("dividing Intervals");
        QuadDebug(cudaMalloc((void**)&genRegions,
                             sizeof(T) * numActiveRegions * NDIM *
                               numOfDivisionOnDimension));
        QuadDebug(cudaMalloc((void**)&genRegionsLength,
                             sizeof(T) * numActiveRegions * NDIM *
                               numOfDivisionOnDimension));
        CudaCheckError();

        divideIntervalsGPU<T>
          <<<numBlocks, numThreads>>>(NDIM,
                                      genRegions,
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
        //nvtxRangePop();
        cudaDeviceSynchronize();
        CudaCheckError();
      } else {
        numRegions = 0;
      }

      return numInActiveRegions;
    }

    void
    GetNextThreshold(T min, T max, int rightDirection, T& current)
    {
      if (rightDirection) {
        T diff = abs(max - current);
        current += diff * .5f;
        // printf("right direction diff:%.15e min:%.15e max:%.15e\n", diff, min,
        // max);
      } else {
        T diff = abs(min - current);
        current -= diff * .5f;
        // printf("left direction diff:%.15e min:%.15e max:%.15e\n", diff, min,
        // max);
      }
    }

    bool
    AdjustErrorThreshold(T ErrorestMarkedFinished,
                         T MaxErrorestToAllowedToFinish,
                         T percentRegionsToKeep,
                         int& currentDirection,
                         T& lastThreshold,
                         T& threshold,
                         T& minThreshold,
                         T& maxThreshold,
                         int& numDirectionChanges)
    {
      // printf("adjusting threshold with min max:%.15e, %.15e\n", minThreshold,
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
      // printf("new from within adjustment min max:%.15e, %.15e\n",
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
               T& integral,
               T& error,
               size_t& nregions,
               T iterEstimate,
               T iterErrorest,
               T iter_finished_estimate,
               T iter_finished_errorest,
               T leaves_estimate,
               T leaves_errorest,
               T epsrel,
               T epsabs,
               int iteration)
    {

      int requiredDigits =
        ceil(log10(1 / epsrel)); // this is no longer used in RelErrClassify,
                                 // move below when used by Filter
      estimateHasConverged =
        estimateHasConverged == false ?
          //(iteration >= 2 ?
          (iteration >= 10 ?
             sigDigitsSame(
               lastAvg, secondTolastAvg, leaves_estimate, requiredDigits) :
             false) :
          true;

      secondTolastAvg = lastAvg;
      lastAvg = leaves_estimate;
      lastErr = leaves_errorest;

      T mem_need_have_ratio =
        (T)GetGPUMemNeededForNextIteration_CallBeforeSplit() /
        ((T)Device.GetAmountFreeMem());
      bool enoughMemForNextIter = mem_need_have_ratio < 1.;

      if (enoughMemForNextIter &&
          !estimateHasConverged) // don't filter if we haven't converged and we
                                 // have enough mem
        return;

      if (mem_need_have_ratio < .1)
        return;

      T targetError = abs(leaves_estimate) * epsrel;
      size_t numThreads = BLOCK_SIZE;

      size_t numBlocks =
        numRegions / numThreads + ((numRegions % numThreads) ? 1 : 0);

      T MaxPercentOfErrorBudget = .25;
      T acceptableThreshold = 0.;
      int* unpolishedRegions = 0;
      int* scannedArray = 0; // de-allocated at the end of this function
      size_t targetRegionNum = numRegions / 2;

      T ErrThreshold = iterErrorest / (numRegions); // starts out as avg value

      T lastThreshold = ErrThreshold;
      size_t numActiveRegions = numRegions;
      T iter_polished_errorest = 0.;
      T iter_polished_estimate = 0.;
      int numDirectionChanges = 0;

      QuadDebug(Device.AllocateMemory((void**)&unpolishedRegions,
                                      sizeof(int) * numRegions));

      thrust::device_ptr<int> d_ptr =
        thrust::device_pointer_cast(unpolishedRegions);
      int direction = -1;

      thrust::device_ptr<T> d_ptrE = thrust::device_pointer_cast(dRegionsError);

      thrust::pair<typename thrust::device_vector<T>::iterator,
                   typename thrust::device_vector<T>::iterator>
        tuple;
      auto __tuple = thrust::minmax_element(d_ptrE, d_ptrE + numRegions);
      T min = *__tuple.first;
      T max = *__tuple.second;
      thrust::device_ptr<T> wrapped_ptr;
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
        // printf("trying threshold:%.15f\n", ErrThreshold);
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
        // printf("number active regions:%lu with potential threshold %.15e
        // current regions:%lu target:%lu\n", numActiveRegions, ErrThreshold,
        // numRegions, targetRegionNum);
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
            // printf("Found it %.15f +- %.15f\n", integral, error);
            break;
          } else if (iter_polished_errorest <= .95 * (targetError - error)) {
            acceptableThreshold = ErrThreshold;
            // printf("Found acceptableThreshold:%.15e\n", acceptableThreshold);
          }
        }
        T unpolishedPercentage = (T)(numActiveRegions) / (T)numRegions;
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
        mustFinish = true;
        QuadDebug(Device.ReleaseMemory(unpolishedRegions));
        // printf("must finish triggered\n");
      } else {
        // printf("worked now have %lu active regions\n", numActiveRegions);
        QuadDebug(Device.ReleaseMemory(activeRegions));
        activeRegions = unpolishedRegions;
        // CudaCheckError();
      }
      CudaCheckError();
    }

    void
    AllocVolArrays(Volume<T, NDIM> const* vol)
    {
      /*
        this is invoked by IntegateFirstPhase and doesn't need to called by user
        is currently present in the test RegionSampling due to ~Kernel's
        destruction of highs, lows if AllocVolArrays is never called, cudaFree
        will cause error
      */
      // //nvtxRangePush("AllocVolArrays");
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
      // //nvtxRangePop();
    }

    void
    RelErrClassify(int* activeRegions, size_t nregions, T epsrel, int iteration)
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
        Device.AllocateMemory((void**)&newErrs, sizeof(T) * numRegions));
      // printf("Refine Err Relerr classify %lu regions\n", numRegions);
      RefineError<T><<<numBlocks, BLOCK_SIZE>>>(dRegionsIntegral,
                                                dRegionsError,
                                                dParentsIntegral,
                                                dParentsError,
                                                newErrs,
                                                activeRegions,
                                                numRegions,
                                                epsrel,
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
    doubleToString(T val, int prec_level)
    {
      std::ostringstream out;
      out.precision(prec_level);
      out << std::fixed << val;
      return out.str();
    }

    bool
    sigDigitsSame(T x, T y, T z, int requiredDigits)
    {
      T third = abs(x);
      T second = abs(y);
      T first = abs(z);

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
    IterationAllocations(T*& dRegionsIntegral,
                         T*& dRegionsError,
                         T*& dParentsIntegral,
                         T*& dParentsError,
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

    T
    ComputeIterContribution(T* estimates)
    {
      thrust::device_ptr<T> wrapped_ptr =
        thrust::device_pointer_cast(estimates);
      return thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);
    }

    void
    FixErrorBudgetOverflow(int* activeRegions,
                           T& integral,
                           T& error,
                           T& iter_finished_estimate,
                           T& iter_finished_errorest,
                           T newLeavesEstimate,
                           T epsrel)
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
    ComputeFinishedEstimates(T& iter_finished_estimate,
                             T& iter_finished_errorest,
                             T* dRegionsIntegral,
                             T iter_estimate,
                             T* dRegionsError,
                             T iter_errorest,
                             int* activeRegions)
    {
      thrust::device_ptr<int> wrapped_mask;
      // printf("1st Inner dot product %lu regions\n", numRegions);
      // //nvtxRangePush("Inner Product 1");
      wrapped_mask = thrust::device_pointer_cast(activeRegions);

      thrust::device_ptr<T> wrapped_ptr =
        thrust::device_pointer_cast(dRegionsIntegral);
      iter_finished_estimate =
        iter_estimate - thrust::inner_product(thrust::device,
                                              wrapped_ptr,
                                              wrapped_ptr + numRegions,
                                              wrapped_mask,
                                              0.);
      // //nvtxRangePop();
      // printf("2nd Inner dot product %lu regions\n", numRegions);
      // //nvtxRangePush("Inner Product 2");
      wrapped_ptr = thrust::device_pointer_cast(dRegionsError);
      iter_finished_errorest =
        iter_errorest - thrust::inner_product(thrust::device,
                                              wrapped_ptr,
                                              wrapped_ptr + numRegions,
                                              wrapped_mask,
                                              0.);
      // //nvtxRangePop();
    }

    void
    CheckZeroNumRegionsTermination(T& integral,
                                   T& error,
                                   T leaves_estimate,
                                   T leaves_errorest)
    {
      if (numRegions == 0) {
        integral = leaves_estimate;
        error = leaves_errorest;
      }
    }

    bool
    CheckTerminationCondition(T leaves_estimate,
                              T leaves_errorest,
                              T& integral,
                              T& error,
                              size_t& nregions,
                              T epsrel,
                              T epsabs,
                              int iteration,
                              int* activeRegions,
                              int* subDividingDimension)
    {
      if (std::isnan(integral) || std::isnan(error)) {
        fail = 0;
        return true;
      }

      if ((iteration != 0 &&
           leaves_errorest <= MaxErr(leaves_estimate, epsrel, epsabs)) ||
          mustFinish) {

        integral = leaves_estimate;
        error = leaves_errorest;
        nregions += numRegions;

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
                        Volume<T, NDIM> const* vol,
                        int last_iteration = 0)
    {
      size_t numThreads = BLOCK_SIZE;
      size_t numBlocks = numRegions;
      // printf("Amount of free memory at start of iteration:%lu\n",
      // Device.GetAmountFreeMem());

      // //nvtxRangePush("Iteration Allocations");
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
      // //nvtxRangePop();
	 set_device_array<int>(activeRegions, numRegions, 1.);

      /*if(iteration == 14){
          printf("about to display\n");
          display<double, 2>(numRegions*NDIM, dRegions, dRegionsLength);
      }*/

      // //nvtxRangePush("INTEGRATE_GPU_PHASE1");
      quad::Func_Evals<NDIM> fevals;
      bool constexpr debug = false;
      quad::INTEGRATE_GPU_PHASE1<IntegT, T, NDIM, BLOCK_SIZE, debug>
        <<<numBlocks, numThreads>>>(
          d_integrand,
          dRegions,
          dRegionsLength,
          numRegions,
          dRegionsIntegral,
          dRegionsError,
          subDividingDimension,
          epsrel,
          epsabs,
          constMem,
          //rule.GET_FEVAL(),
          //rule.GET_NSETS(),
          lows,
          highs,
          generators,
		  fevals);

      neval += numRegions * fEvalPerRegion;
      cudaDeviceSynchronize();
      CudaCheckError();
      // //nvtxRangePop();
      // printf("Reduction 1 %lu regions\n", numRegions);
      // //nvtxRangePush("Reduction 1");
      T iter_estimate = ComputeIterContribution(dRegionsIntegral);
      // //nvtxRangePop();
      T leaves_estimate = integral + iter_estimate;
      //nvtxRangePush("Rel Error Classify");
      RelErrClassify(activeRegions, nregions, epsrel, iteration);
      //nvtxRangePop();
      // printf("Reduction 2 %lu regions\n", numRegions);
      T iter_finished_estimate = 0, iter_finished_errorest = 0;
      // //nvtxRangePush("Reduction 2");
      T iter_errorest = ComputeIterContribution(dRegionsError);
      // //nvtxRangePop();
      T leaves_errorest = error + iter_errorest;

      ComputeFinishedEstimates(iter_finished_estimate,
                               iter_finished_errorest,
                               dRegionsIntegral,
                               iter_estimate,
                               dRegionsError,
                               iter_errorest,
                               activeRegions);
      integral += iter_finished_estimate;
      error += iter_finished_errorest;
      Phase_I_PrintFile(vol,
                        numRegions,
                        activeRegions,
                        leaves_estimate,
                        leaves_errorest,
                        iter_estimate,
                        iter_errorest,
                        iter_finished_estimate,
                        iter_finished_errorest,
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
      // printf("%i, iter estimates: %.15e, %.15e (%.15e +-
      // %.15e),numRegions:%lu\n", iteration, iter_estimate, iter_errorest,
      // iter_finished_estimate, iter_finished_errorest, numRegions);
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

      //nvtxRangePush("HS CLASSIFY");
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
      //nvtxRangePop();
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
      if (iteration < 700 && fail == 1) {
        //nvtxRangePush("GenerateActiveIntervals");
        size_t numInActiveIntervals =
          GenerateActiveIntervals(activeRegions,
                                  subDividingDimension,
                                  dRegionsIntegral,
                                  dRegionsError,
                                  dParentsIntegral,
                                  dParentsError);
        CheckZeroNumRegionsTermination(
          integral, error, leaves_estimate, leaves_errorest);
        nregions += numInActiveIntervals;
        nFinishedRegions += numInActiveIntervals;
      } else {
        nregions += numRegions;
      }
      //nvtxRangePop();
      QuadDebug(cudaFree(activeRegions));
      QuadDebug(cudaFree(subDividingDimension));
    }

    /*template <typename IntegT>
    void
    EvaluateAtCuhrePoints(IntegT* d_integrand,
                          VerboseResults& resultsObj,
                          Volume<T, NDIM>* vol = nullptr)
    {

      size_t numRegions = 1;
      size_t numBlocks = numRegions;
      size_t numThreads = BLOCK_SIZE;
      double epsrel = 1.e-3;
      double epsabs = 1.e-22;

      double* funcEvals =
        quad::cuda_malloc_managed<double>(NDIM * rule.GET_FEVAL());
      double* results = quad::cuda_malloc_managed<double>(rule.GET_FEVAL());

      QuadDebug(Device.AllocateMemory((void**)&generators,
                                      sizeof(double) * NDIM * fEvalPerRegion));
      ComputeGenerators<double, NDIM>
        <<<1, BLOCK_SIZE>>>(generators, fEvalPerRegion, constMem);
      cudaDeviceSynchronize();
      CudaCheckError();
      AllocVolArrays(vol);

      int iteration = 0;
      dRegionsError = nullptr, dRegionsIntegral = nullptr;
      int *activeRegions = nullptr, *subDividingDimension = nullptr;

      IterationAllocations(dRegionsIntegral,
                           dRegionsError,
                           dParentsIntegral,
                           dParentsError,
                           activeRegions,
                           subDividingDimension,
                           iteration);

      gEvaluateAtCuhrePoints<IntegT, T, NDIM, BLOCK_SIZE>
        <<<numBlocks, numThreads>>>(d_integrand,
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
                                    generators,
                                    results,
                                    funcEvals);

      cudaDeviceSynchronize();

      resultsObj.numFuncEvals = rule.GET_FEVAL();
      resultsObj.results.reserve(rule.GET_FEVAL());

      for (int i = 0; i < rule.GET_FEVAL(); ++i) {
        resultsObj.results[i] = results[i];
      }

      for (int i = 0; i < rule.GET_FEVAL(); ++i) {
        std::vector<double> evalPoints;
        for (int dim = 0; dim < NDIM; ++dim) {
          evalPoints.push_back(funcEvals[i * NDIM + dim]);
        }
        resultsObj.funcEvaluationPoints.push_back(evalPoints);
      }
    }*/

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
                        Volume<T, NDIM> const* vol = nullptr)
    {

      QuadDebug(Device.AllocateMemory((void**)&generators,
                                      sizeof(T) * NDIM * fEvalPerRegion));
      CudaCheckError();
      ComputeGenerators<T, NDIM>
        <<<1, BLOCK_SIZE>>>(generators, fEvalPerRegion, constMem);
      cudaDeviceSynchronize();
      CudaCheckError();

      AllocVolArrays(vol);
      CudaCheckError();
      PrintOutfileHeaders();
      int lastIteration = 0;
      int iteration = 0;
      fail = 1;

      for (iteration = 0; iteration < 700 && fail == 1 && mustFinish == false;
           iteration++) {
        CudaCheckError();
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
        CudaCheckError();
      }

      CudaCheckError();

      StringstreamToFile(finishedOutfile.str(), phase1out.str(), outLevel);
      QuadDebug(Device.ReleaseMemory(dRegions));
      CudaCheckError();
      QuadDebug(Device.ReleaseMemory(dRegionsLength));
      CudaCheckError();
      QuadDebug(Device.ReleaseMemory(dParentsIntegral));
      CudaCheckError();
      QuadDebug(Device.ReleaseMemory(dParentsError));
      CudaCheckError();
      QuadDebug(Device.ReleaseMemory(lows));
      CudaCheckError();
      QuadDebug(Device.ReleaseMemory(highs));
      CudaCheckError();
      QuadDebug(cudaFree(generators));
      CudaCheckError();

      bool convergence = false;
      convergence = error <= MaxErr(integral, epsrel, epsabs);
      return !convergence;
    }
  };

}
#endif
