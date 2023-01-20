#ifndef CUDACUHRE_QUAD_GPUQUAD_KERNEL_CUH
#define CUDACUHRE_QUAD_GPUQUAD_KERNEL_CUH

#include "common/cuda/Volume.cuh"
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
#include "common/cuda/cudaMemoryUtil.h"


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
}
#endif
