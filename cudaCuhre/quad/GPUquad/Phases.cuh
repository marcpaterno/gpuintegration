#ifndef CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH
#define CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH

#include "../util/Volume.cuh"
#include "Sample.cuh"
#include <cooperative_groups.h>

#define FINAL 0
// added for variadic cuprintf
#include <stdarg.h>
#include <stdio.h>
namespace quad {

  __device__ void
  cuprintf(const char* fmt, ...)
  {
    va_list args;
    va_start(args, fmt);

    if (threadIdx.x == 0) {
      while (*fmt != '\0') {
        if (*fmt == 'd') {
          int i = va_arg(args, int);
          printf("%d\n", i);
        } else if (*fmt == 'c') {
          int c = va_arg(args, int);
          printf("%c\n", c);
        } else if (*fmt == 'f') {
          double d = va_arg(args, double);
          printf("%f\n", d);
        }
        ++fmt;
      }
    }
    va_end(args);
  }

  __device__ __host__ double
  ScaleValue(double val, double min, double max)
  {
    // assert that max > min
    double range = fabs(max - min);
    return min + val * range;
  }

  template <typename T, int NDIM>
  __global__ void
  QuickMassSample(T* dRegions,
                  T* dRegionsLength,
                  size_t numRegions,
                  Region<NDIM> sRegionPool[],
                  T* dRegionsIntegral,
                  T* dRegionsError,
                  Structures<T> constMem,
                  int FEVAL,
                  int NSETS)
  {
    T ERR = 0, RESULT = 0;
    INIT_REGION_POOL(
      dRegions, dRegionsLength, numRegions, &constMem, FEVAL, NSETS);

    if (threadIdx.x == 0) {
      dRegionsIntegral[blockIdx.x] = sRegionPool[threadIdx.x].result.avg;
      dRegionsError[blockIdx.x] = sRegionPool[threadIdx.x].result.err;
      __syncthreads();
    }
  }

  template <typename T>
  __global__ void
  RefineError(T* dRegionsIntegral,
              T* dRegionsError,
              T* dParentsIntegral,
              T* dParentsError,
              T* newErrs,
              int* activeRegions,
              size_t currIterRegions,
			  //double finished_estimate,
			  double finished_errorest,
			  //double queued_estimate,
			  double queued_errorest,
			  double last_it_estimate,
			  //double last_it_errorest,
			  //size_t nregions,
			  //size_t numFinishedRegions,
              T epsrel,
              T epsabs,
              int iteration,
			  bool estConverged)
  {
	//can we do anythign with the rest of the threads? maybe launch more blocks instead and a  single thread per block?
    if (threadIdx.x == 0 && blockIdx.x < currIterRegions) {
      int fail = 0;

      T selfErr = dRegionsError[blockIdx.x + currIterRegions];
      T selfRes = dRegionsIntegral[blockIdx.x + currIterRegions];
		
      // that's how indices to the right to find the sibling
      // but we want the sibling to be found at the second half of the array
      // only, to avoid race conditions

      int siblingIndex = (currIterRegions / 2) + blockIdx.x;
      if (siblingIndex < currIterRegions) {
        siblingIndex += currIterRegions;
      }

      T siblErr = dRegionsError[siblingIndex];
      T siblRes = dRegionsIntegral[siblingIndex];
		
      T parRes = dParentsIntegral[blockIdx.x];
	
      T diff = siblRes + selfRes - parRes;
      diff = fabs(.25 * diff);

      T err = selfErr + siblErr;

      if (err > 0.0) {
        T c = 1 + 2 * diff / err;
        selfErr *= c;
      }
	  
	  selfErr += diff;
	  
	  auto isPolished = [last_it_estimate, 
						 currIterRegions, 
						 finished_errorest, 
						 queued_errorest,
						 epsrel, 
						 epsabs,
						 estConverged](double selfRes, double selfErr)
	  {
		bool minIterReached = estConverged;
       // bool minIterReached = true;
		double GlobalErrTarget = fabs(last_it_estimate)*epsrel;
		double remainGlobalErrRoom = GlobalErrTarget - finished_errorest - queued_errorest; 
		//bool worstCaseScenarioGood =  selfErr*currIterRegions < .25*remainGlobalErrRoom;
        bool worstCaseScenarioGood =  selfRes < last_it_estimate*epsrel;
		bool selfErrTarget = fabs(selfRes)*epsrel;
		bool verdict = worstCaseScenarioGood && minIterReached;
        //if(verdict == true && (selfErr / (fabs(selfRes)*epsrel)) > 1.)
        //    printf("%e +- %e remainGlobalErrRoom:%e GlobalErrTarget:%e queued_errorest:%e finished_errorest:%e\n", selfRes, selfErr, remainGlobalErrRoom, GlobalErrTarget, queued_errorest, finished_errorest);
		return verdict;
	  };
	       
	 if (/*isPolished(selfRes, selfErr) == true ||*/ selfErr / (fabs(selfRes)*epsrel) < 1.) {
           // printf("%e +- %e r:%f\n", selfErr, selfRes, (selfErr / (fabs(selfRes)*epsrel)));
        newErrs[blockIdx.x] = selfErr;
      } else {
        fail = 1;
        newErrs[blockIdx.x] = 0;
        dRegionsIntegral[blockIdx.x] = 0;
      }
	  
      activeRegions[blockIdx.x] = fail;
      newErrs[blockIdx.x + currIterRegions] = selfErr;
    }
  }

  template <typename IntegT, typename T, int NDIM>
  __device__ void
  INIT_REGION_POOL(IntegT* d_integrand,
                   T* dRegions,
                   T* dRegionsLength,
                   size_t numRegions,
                   const Structures<T>& constMem,
                   int FEVAL,
                   int NSETS,
                   Region<NDIM> sRegionPool[],
                   T* lows,
                   T* highs,
				   int iteration,
                   int depth)
  {
    size_t index = blockIdx.x;

    if (threadIdx.x == 0) {
      for (int dim = 0; dim < NDIM; ++dim) {
        T lower = dRegions[dim * numRegions + index];
        sRegionPool[threadIdx.x].bounds[dim].lower = lower;
        sRegionPool[threadIdx.x].bounds[dim].upper =
          lower + dRegionsLength[dim * numRegions + index];

        sBound[dim].unScaledLower = lows[dim];
        sBound[dim].unScaledUpper = highs[dim];
		
        if (sRegionPool[threadIdx.x].bounds[dim].lower ==
            sRegionPool[threadIdx.x].bounds[dim].upper /*|| (iteration == 23 && blockIdx.x == 0)*/)
          printf("sRegionPool [%i] bounds[%i]: %.15f - %.15f\n",
                 blockIdx.x,
                 dim,
                 sRegionPool[threadIdx.x].bounds[dim].lower,
                 sRegionPool[threadIdx.x].bounds[dim].upper);
        if (sBound[dim].unScaledLower == sBound[dim].unScaledUpper)
          printf("serror [%i](%i) sbounds[%i]: %.15f - %.15f\n",
                 blockIdx.x,
                 threadIdx.x,
                 dim,
                 sBound[dim].unScaledLower,
                 sBound[dim].unScaledUpper);
        sRegionPool[threadIdx.x].div = 
          depth; // carry info over or compute it?
      }
    }

    __syncthreads();
    SampleRegionBlock<IntegT, T, NDIM>(
      d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool);
    __syncthreads();
  }

  template <typename IntegT, typename T, int NDIM>
  __global__ void
  INTEGRATE_GPU_PHASE1(IntegT* d_integrand,
                       T* dRegions,
                       T* dRegionsLength,
                       size_t numRegions,
                       T* dRegionsIntegral,
                       T* dRegionsError,
                       int* activeRegions,
                       int* subDividingDimension,
                       T epsrel,
                       T epsabs,
                       Structures<T> constMem,
                       int FEVAL,
                       int NSETS,
                       T* lows,
                       T* highs,
					   int iteration,
                       int depth)
  {
    __shared__ Region<NDIM> sRegionPool[SM_REGION_POOL_SIZE];
    __shared__ T shighs[NDIM];
    __shared__ T slows[NDIM];

    if (threadIdx.x == 0) {
      for (int i = 0; i < NDIM; ++i) {
        slows[i] = lows[i];
        shighs[i] = highs[i];
      }
    }

    T ERR = 0, RESULT = 0;
    int fail = 1;

    INIT_REGION_POOL<IntegT>(d_integrand,
                             dRegions,
                             dRegionsLength,
                             numRegions,
                             constMem,
                             FEVAL,
                             NSETS,
                             sRegionPool,
                             slows,
                             shighs,
							 iteration,
                             depth);

    if (threadIdx.x == 0) {
      ERR = sRegionPool[threadIdx.x].result.err;
      RESULT = sRegionPool[threadIdx.x].result.avg;

      T ratio = ERR / MaxErr(RESULT, epsrel, epsabs);
      int fourthDiffDim = sRegionPool[threadIdx.x].result.bisectdim;
      dRegionsIntegral[gridDim.x + blockIdx.x] = RESULT;
	  dRegionsError[gridDim.x + blockIdx.x] = ERR;
	
	  /*if(iteration == 23 && blockIdx.x == 0)
		  printf("bounds and length numRegions:%lu %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", numRegions, 
																			dRegions[0*numRegions + blockIdx.x], dRegionsLength[0*numRegions + blockIdx.x],
																			dRegions[1*numRegions + blockIdx.x], dRegionsLength[1*numRegions + blockIdx.x],
																			dRegions[2*numRegions + blockIdx.x], dRegionsLength[2*numRegions + blockIdx.x],
																			dRegions[3*numRegions + blockIdx.x], dRegionsLength[3*numRegions + blockIdx.x],
																			dRegions[4*numRegions + blockIdx.x], dRegionsLength[4*numRegions + blockIdx.x],
																			dRegions[5*numRegions + blockIdx.x], dRegionsLength[5*numRegions + blockIdx.x],
																			dRegions[6*numRegions + blockIdx.x], dRegionsLength[6*numRegions + blockIdx.x],
																			dRegions[7*numRegions + blockIdx.x], dRegionsLength[7*numRegions + blockIdx.x]);*/
      // if (ratio > 1) {
      // fail = 1;
      //}

      activeRegions[blockIdx.x] = fail;
      subDividingDimension[blockIdx.x] = fourthDiffDim;
      dRegionsIntegral[blockIdx.x] = RESULT;
      dRegionsError[blockIdx.x] = ERR;
	  
      if (/*ratio > 1 && */numRegions == 1) {//do we need to zero it out already?
        dRegionsIntegral[blockIdx.x] = 0;
        dRegionsError[blockIdx.x] = 0;
      }
    }
  }

  ////PHASE 2 Procedures Starts
  template <typename T, int NDIM>
  __device__ void
  ComputeErrResult(T& ERR, T& RESULT, Region<NDIM> sRegionPool[])
  {
    if (threadIdx.x == 0) {
      ERR = sRegionPool[threadIdx.x].result.err;
      RESULT = sRegionPool[threadIdx.x].result.avg;
    }
    __syncthreads();
  }

  /*
          initializes shared memory with empty regions
  */
  template <int NDIM>
  __device__ int
  InitSMemRegions(Region<NDIM> sRegionPool[])
  {
    int idx = 0;
    for (; idx < SM_REGION_POOL_SIZE / BLOCK_SIZE; ++idx) {

      int index = idx * BLOCK_SIZE + threadIdx.x;
      sRegionPool[index].div = 0;
      sRegionPool[index].result.err = 0;
      sRegionPool[index].result.avg = 0;
      sRegionPool[index].result.bisectdim = 0;

      for (int dim = 0; dim < NDIM; ++dim) {
        sRegionPool[index].bounds[dim].lower = 0;
        sRegionPool[index].bounds[dim].upper = 0;
      }
    }

    int index = idx * BLOCK_SIZE + threadIdx.x;
    if (index < SM_REGION_POOL_SIZE) {

      sRegionPool[index].div = 0;
      sRegionPool[index].result.err = 0;
      sRegionPool[index].result.avg = 0;
      sRegionPool[index].result.bisectdim = 0;

      for (int dim = 0; dim < NDIM; ++dim) {
        sRegionPool[index].bounds[dim].lower = 0;
        sRegionPool[index].bounds[dim].upper = 0;
      }
    }
    return 1;
  }

  /*
          sets the bounds of the 1st region from 0 to 1 while the
          global bounds(1st region's real boundaries) are assigned to sBound
          every time it is called it resets the global region list size to
                (SM_REGION_POOL_SIZE / 2)
  */

  __device__ size_t
  GetSiblingIndex(size_t numRegions)
  {
    return (2 * blockIdx.x / numRegions) < 1 ? blockIdx.x + numRegions :
                                               blockIdx.x - numRegions;
  }

  template <typename T, int NDIM>
  __device__ int
  SET_FIRST_SHARED_MEM_REGION(Region<NDIM> sRegionPool[],
                              const T* dRegions,
                              const T* dRegionsLength,
                              const T* lows,
                              const T* highs,
                              size_t numRegions,
                              size_t blockIndex)
  {
    size_t intervalIndex = blockIndex;

    if (threadIdx.x == 0) {
      gRegionPoolSize = (SM_REGION_POOL_SIZE / 2);
      for (int dim = 0; dim < NDIM; ++dim) {
        sRegionPool[threadIdx.x].bounds[dim].lower = 0;
        sRegionPool[threadIdx.x].bounds[dim].upper = 1;

        T lower = dRegions[dim * numRegions + intervalIndex];
        T upper = lower + dRegionsLength[dim * numRegions + intervalIndex];

        sBound[dim].unScaledLower = ScaleValue(lower, lows[dim], highs[dim]);
        sBound[dim].unScaledUpper = ScaleValue(upper, lows[dim], highs[dim]);
      }
    }
    __syncthreads();
    return 1;
  }

  template <typename T, int NDIM>
  __device__ int
  set_first_shared_mem_region(Region<NDIM> sRegionPool[],
                              T* lows,
                              T* highs,
                              size_t numRegions,
                              size_t blockIndex)
  {
    if (threadIdx.x == 0) {
      gRegionPoolSize = (SM_REGION_POOL_SIZE / 2);
      for (int dim = 0; dim < NDIM; ++dim) {
        sRegionPool[threadIdx.x].bounds[dim].lower = 0;
        sRegionPool[threadIdx.x].bounds[dim].upper = 1;
        sBound[dim].unScaledLower = lows[dim];
        sBound[dim].unScaledUpper = highs[dim];
      }
    }
    __syncthreads();
    return 1;
  }

  template <typename T, int NDIM>
  __device__ int
  set_first_shared_mem_region(Region<NDIM> sRegionPool[],
                              Region<NDIM>* ggRegionPool,
                              T* lows,
                              T* highs,
                              size_t numRegions,
                              size_t blockIndex)
  {
    size_t intervalIndex = blockIndex;
    if (threadIdx.x == 0) {
      gRegionPoolSize =
        (SM_REGION_POOL_SIZE / 2); // why start with 64 as gRegionPoolSize

      for (int dim = 0; dim < NDIM; ++dim) {
        sRegionPool[threadIdx.x].bounds[dim].lower = 0;
        sRegionPool[threadIdx.x].bounds[dim].upper = 1;
        T lower = ggRegionPool[intervalIndex].bounds[dim].lower;
        T upper = ggRegionPool[intervalIndex].bounds[dim].upper;

        sBound[dim].unScaledLower = ScaleValue(lower, lows[dim], highs[dim]);
        sBound[dim].unScaledUpper = ScaleValue(upper, lows[dim], highs[dim]);
      }
    }
    return 1;
  }

  template <typename IntegT, typename T, int NDIM>
  __device__ void
  ALIGN_GLOBAL_TO_SHARED(Region<NDIM> sRegionPool[], Region<NDIM>*& gPool)
  {

    int idx = 0;
    int index = idx * BLOCK_SIZE + threadIdx.x;
    __syncthreads();

    for (idx = 0; idx < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE; ++idx) {
      size_t index = idx * BLOCK_SIZE + threadIdx.x;
      gRegionPos[index] = index;
      gPool[index] = sRegionPool[index];
    }

    index = idx * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      gRegionPos[index] = index;
      gPool[index] = sRegionPool[index];
    }
  }

  template <class T>
  __device__ void
  swap(T& a, T& b)
  {
    T c(a);
    a = b;
    b = c;
  }

  template <typename T, int NDIM>
  __device__ void
  INSERT_GLOBAL_STORE(const Region<NDIM>* const sRegionPool,
                      int gpuId,
                      Region<NDIM>* const gPool)
  {
    __syncthreads();

    int iterationsPerThread = 0;
    for (iterationsPerThread = 0;
         iterationsPerThread < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE;
         ++iterationsPerThread) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      gPool[gRegionPos[index]] = sRegionPool[index];
      gPool[gRegionPoolSize + index] =
        sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
    }

    __syncthreads();
    int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      gPool[gRegionPos[index]] = sRegionPool[index];
      gPool[gRegionPoolSize + index] =
        sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      gRegionPoolSize = gRegionPoolSize + (SM_REGION_POOL_SIZE / 2);
    }
    __syncthreads();
  }

  template <typename T, int NDIM>
  __device__ void
  INSERT_GLOBAL_STORE2(Region<NDIM>* sRegionPool,
                       Region<NDIM>* gRegionPool,
                       int gpuId,
                       Region<NDIM>* gPool,
                       int sRegionPoolSize)
  {
    // size_t startIndex = blockIdx.x * 2048;
    __syncthreads();

    // Copy existing global regions into newly allocated spaced

    int iterationsPerThread = 0;

    for (iterationsPerThread = 0;
         iterationsPerThread < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE;
         ++iterationsPerThread) {
      size_t index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      gPool[gRegionPos[index]] = sRegionPool[index];
      if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize)
        gPool[gRegionPoolSize + index] =
          sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
    }

    __syncthreads();
    size_t index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      gPool[gRegionPos[index]] = sRegionPool[index];

      if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize)
        gPool[gRegionPoolSize + index] =
          sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      gRegionPoolSize = gRegionPoolSize + (SM_REGION_POOL_SIZE / 2);
    }
    __syncthreads();
  }

  template <typename T, int NDIM>
  __device__ void
  insert_global_store(Region<NDIM>* sRegionPool,
                      Region<NDIM>* gRegionPool,
                      int gpuId,
                      Region<NDIM>* gPool,
                      int sRegionPoolSize)
  {

    __syncthreads();

    int iterationsPerThread = 0;

    for (iterationsPerThread = 0;
         iterationsPerThread < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE;
         ++iterationsPerThread) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      gPool[gRegionPos[index]] = sRegionPool[index];

      if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize) {
        gPool[gRegionPoolSize + index] =
          sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
      }

      for (int dim = 0; dim < NDIM; dim++) {
        gPool[gRegionPos[index]].bounds[dim].lower =
          sBound[dim].unScaledLower +
          sRegionPool[index].bounds[dim].lower *
            (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
        gPool[gRegionPos[index]].bounds[dim].upper =
          sBound[dim].unScaledLower +
          sRegionPool[index].bounds[dim].upper *
            (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);

        if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize) {
          gPool[gRegionPoolSize + index].bounds[dim].lower =
            sBound[dim].unScaledLower +
            sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].bounds[dim].lower *
              (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
          gPool[gRegionPoolSize + index].bounds[dim].upper =
            sBound[dim].unScaledLower +
            sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].bounds[dim].upper *
              (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
        }
      }
    }

    __syncthreads();
    int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      gPool[gRegionPos[index]] = sRegionPool[index];

      if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize) {
        gPool[gRegionPoolSize + index] =
          sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
      }

      for (int dim = 0; dim < NDIM; dim++) {
        gPool[gRegionPos[index]].bounds[dim].lower =
          sBound[dim].unScaledLower +
          sRegionPool[index].bounds[dim].lower *
            (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
        gPool[gRegionPos[index]].bounds[dim].upper =
          sBound[dim].unScaledLower +
          sRegionPool[index].bounds[dim].upper *
            (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);

        if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize) {
          gPool[gRegionPoolSize + index].bounds[dim].lower =
            sBound[dim].unScaledLower +
            sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].bounds[dim].lower *
              (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
          gPool[gRegionPoolSize + index].bounds[dim].upper =
            sBound[dim].unScaledLower +
            sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].bounds[dim].upper *
              (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      gRegionPoolSize = gRegionPoolSize + (SM_REGION_POOL_SIZE / 2);
    }
    __syncthreads();
  }

  template <typename T>
  __device__ void
  EXTRACT_MAX(T* serror, size_t* serrorPos, size_t gSize)
  {
    for (size_t offset = gSize / 2; offset > 0; offset >>= 1) {
      int idx = 0;

      for (idx = 0; idx < offset / BLOCK_SIZE; ++idx) {
        size_t index = idx * BLOCK_SIZE + threadIdx.x;
        if (index < offset) {
          if (serror[index] < serror[index + offset]) {
            swap(serror[index], serror[index + offset]);
            swap(serrorPos[index], serrorPos[index + offset]);
          }
        }
      }

      size_t index = idx * BLOCK_SIZE + threadIdx.x;
      if (index < offset) {
        if (serror[index] < serror[index + offset]) {
          swap(serror[index], serror[index + offset]);
          swap(serrorPos[index], serrorPos[index + offset]);
        }
      }
      __syncthreads();
    }
  }

  template <typename T, int NDIM>
  __device__ void
  EXTRACT_TOPK(Region<NDIM>* sRegionPool,
               Region<NDIM>* gRegionPool,
               Region<NDIM>* gPool)
  {

    T* sarray = (T*)&sRegionPool[0];

    if (threadIdx.x == 0) {
      if ((gRegionPoolSize * sizeof(T) + gRegionPoolSize * sizeof(size_t)) <
          sizeof(Region<NDIM>) * SM_REGION_POOL_SIZE) {
        serror = &sarray[0];
        // TODO:Size of sRegionPool vs sarray constrain
        serrorPos = (size_t*)&sarray[gRegionPoolSize];
      } else {
        serror = (T*)malloc(sizeof(T) * gRegionPoolSize);
        serrorPos = (size_t*)malloc(sizeof(size_t) * gRegionPoolSize);
      }
    }
    __syncthreads();

    int offset = 0;
    // TODO: CHECK EFFECT ON PERFORMANCE IF ANY
    // for (offset = 0; (offset < MAX_GLOBALPOOL_SIZE / BLOCK_SIZE) && (offset <
    // gRegionPoolSize / BLOCK_SIZE); offset++) {
    for (offset = 0; offset < gRegionPoolSize / BLOCK_SIZE; offset++) {
      size_t regionIndex = offset * BLOCK_SIZE + threadIdx.x;
      serror[regionIndex] = gPool[regionIndex].result.err;
      serrorPos[regionIndex] = regionIndex;
    }

    size_t regionIndex = offset * BLOCK_SIZE + threadIdx.x;
    if (regionIndex < gRegionPoolSize) {
      serror[regionIndex] = gPool[regionIndex].result.err;
      serrorPos[regionIndex] = regionIndex;
    }

    __syncthreads();
    for (int k = 0; k < (SM_REGION_POOL_SIZE / 2); ++k) {
      EXTRACT_MAX<T>(&serror[k], &serrorPos[k], gRegionPoolSize - k);
    }

    int iterationsPerThread = 0;
    for (iterationsPerThread = 0;
         iterationsPerThread < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE;
         ++iterationsPerThread) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      size_t pos = serrorPos[index];
      gRegionPos[index] = pos;
    }

    int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      size_t pos = serrorPos[index];
      gRegionPos[index] = pos;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      if (2 * gRegionPoolSize * sizeof(T) >=
          sizeof(Region<NDIM>) * SM_REGION_POOL_SIZE) {
        free(serror);
        free(serrorPos);
      }
    }
    __syncthreads();

    // Copy top K into SM and reset the remaining
    for (iterationsPerThread = 0;
         iterationsPerThread < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE;
         ++iterationsPerThread) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      sRegionPool[index] = gPool[gRegionPos[index]];
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.err = 0;
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.avg = 0;
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].div = 0;
    }

    index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      sRegionPool[index] = gPool[gRegionPos[index]];
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.err = 0;
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.avg = 0;
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].div = 0;
    }
  }

  template <typename T, int NDIM>
  __device__ size_t
  EXTRACT_MAX(Region<NDIM>* sRegionPool,
              Region<NDIM>*& gRegionPool,
              size_t sSize,
              int gpuId,
              Region<NDIM>*& gPool)
  {
    // If array  for regions in shared is full, clear it and update the global
    // memory array
    if (sSize == SM_REGION_POOL_SIZE) {

      INSERT_GLOBAL_STORE<T>(sRegionPool, gpuId, gPool);
      __syncthreads();

      // gRegionPool = gPool;
      EXTRACT_TOPK<T>(sRegionPool, gRegionPool, gPool);
      sSize = (SM_REGION_POOL_SIZE / 2);
      __syncthreads();
    }

    // this checks entire array, but empty values have error of 0, wont' be
    // selected
    for (size_t offset = (SM_REGION_POOL_SIZE / 2); offset > 0; offset >>= 1) {
      int idx = 0;
      for (idx = 0; idx < offset / BLOCK_SIZE; ++idx) {
        size_t index = idx * BLOCK_SIZE + threadIdx.x;
        if (index < offset) {
          Region<NDIM>* r1 = &sRegionPool[index];
          Region<NDIM>* r2 = &sRegionPool[index + offset];
          if (r1->result.err < r2->result.err) {
            swap<Region<NDIM>>(sRegionPool[index], sRegionPool[offset + index]);
          }
        }
      }

      size_t index = idx * BLOCK_SIZE + threadIdx.x;
      if (index < offset) {
        Region<NDIM>* r1 = &sRegionPool[index];
        Region<NDIM>* r2 = &sRegionPool[index + offset];
        if (r1->result.err < r2->result.err) {
          swap<Region<NDIM>>(sRegionPool[index], sRegionPool[offset + index]);
        }
      }
      __syncthreads();
    }

    return sSize;
  }

  template <typename T, int NDIM>
  __device__ void
  SetBlockVars(T* slows,
               T* shighs,
               int& max_global_pool_size,
               Region<NDIM>*& gPool,
               const T* lows,
               const T* highs,
               const int max_regions,
               /*const*/ Region<NDIM>* ggRegionPool)
  {
    if (threadIdx.x == 0) {
      memcpy(slows, lows, sizeof(T) * NDIM);
      memcpy(shighs, highs, sizeof(T) * NDIM);
      max_global_pool_size = max_regions;
      gPool = &ggRegionPool[blockIdx.x * max_regions];
    }
    __syncthreads();
  }

  template <typename T>
  __device__ bool
  ExistsIn(T val, T* array, int array_size)
  {
    for (int i = 0; i < array_size; i++) {
      if (array[i] == val)
        return true;
    }
    return false;
  }

  template <typename IntegT, typename T, int NDIM>
  __global__ void
  BLOCK_INTEGRATE_GPU_PHASE2(IntegT* d_integrand,
                             int* dRegionsNumRegion,
                             T epsrel,
                             T epsabs,
                             int gpuId,
                             Structures<T> constMem,
                             int FEVAL,
                             int NSETS,
                             T* lows,
                             T* highs,
                             int Final,
                             Region<NDIM>* ggRegionPool,
                             T* dParentsIntegral,
                             T* dParentsError,
                             T phase1_lastavg,
                             T phase1_lasterr,
                             T phase1_weightsum,
                             T phase1_avgsum,
                             T* dPh1res,
                             int max_regions,
                             RegionList& batch,
                             int phase1_type = 0,
                             Region<NDIM>* phase1_regs = nullptr,
                             Snapshot<NDIM> snap = Snapshot<NDIM>(),
                             double* global_errorest = nullptr,
                             int* numContributors = nullptr)
  {
    __shared__ Region<NDIM> sRegionPool[SM_REGION_POOL_SIZE];
    __shared__ Region<NDIM>* gPool;
    __shared__ T shighs[NDIM];
    __shared__ T slows[NDIM];
    __shared__ int max_global_pool_size;

    /*
    __shared__ int iterations_without;


    T prev_ratio = 0;
    T prev_error = 0;
    */

    int maxdiv = 0;
    // T origerr;
    // T origavg;
    // int current_snap_array_index = 0;

    // SetBlockVars<T, NDIM>(slows, shighs, max_global_pool_size, gPool, lows,
    // highs, max_regions, ggRegionPool);

    /*
    double personal_estimate_ratio = 0;
    int local_region_cap = 32734;
    */

    if (threadIdx.x == 0) {
      memcpy(slows, lows, sizeof(T) * NDIM);
      memcpy(shighs, highs, sizeof(T) * NDIM);
      max_global_pool_size = max_regions;
      gPool = &ggRegionPool[blockIdx.x * max_regions];
      /*
      iterations_without = 0;
      */
    }

    __syncthreads();              // added for testing
    InitSMemRegions(sRegionPool); // sets every region in shared memory to zero
    int sRegionPoolSize = 1;
    __syncthreads();
    // temporary solution
    if (phase1_type == 0)
      SET_FIRST_SHARED_MEM_REGION(sRegionPool,
                                  batch.dRegions,
                                  batch.dRegionsLength,
                                  slows,
                                  shighs,
                                  batch.numRegions,
                                  blockIdx.x);
    else if (batch.numRegions == 0)
      set_first_shared_mem_region(
        sRegionPool, slows, shighs, batch.numRegions, blockIdx.x);
    else
      set_first_shared_mem_region<T, NDIM>(
        sRegionPool, phase1_regs, slows, shighs, batch.numRegions, blockIdx.x);

    __syncthreads();
    // ERR and sRegionPool[0].result.err are not the same in the beginning
    SampleRegionBlock<IntegT, T, NDIM>(
      d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool);
    ALIGN_GLOBAL_TO_SHARED<IntegT, T, NDIM>(sRegionPool, gPool);
    ComputeErrResult<T, NDIM>(ERR, RESULT, sRegionPool);

    // temporary solution
    if (threadIdx.x == 0 && phase1_type == 0) {
      ERR = batch.dRegionsError[blockIdx.x];
      sRegionPool[0].result.err = ERR;
    } else {
      if (threadIdx.x == 0 && batch.numRegions != 0) {
        ERR = phase1_regs[blockIdx.x].result.err;
        sRegionPool[0].result.err = ERR;
      }
    }

    __syncthreads();
    // if(threadIdx.x == 0)
    //	printf("%i %.20f +- %.20f\n" , blockIdx.x, RESULT, ERR);
    /*
    prev_error = ERR;
    prev_ratio = ERR/MaxErr(RESULT, epsrel, epsabs);

    personal_estimate_ratio = RESULT/dPh1res[0];

    if(numRegions != 0)
            local_region_cap = min(2048,
    (int)(personal_estimate_ratio*2048*numRegions));

    T required_ratio_decrease = abs(1 - prev_ratio)/local_region_cap;
    */

    int nregions = sRegionPoolSize; // is only 1 at this point
    T lastavg = RESULT;
    T lasterr = ERR;
    T weightsum = 1 / fmax(ERR * ERR, ldexp(1., -104));
    T avgsum = weightsum * lastavg;

    T w = 0;
    T avg = 0;
    T sigsq = 0;
    // int snapshot_id = 0;
    while (
      nregions < max_global_pool_size &&
      (ERR >
       MaxErr(RESULT, epsrel, epsabs)) /*&& nregions < local_region_cap*/) {

      sRegionPoolSize =
        EXTRACT_MAX<T, NDIM>(sRegionPool, gPool, sRegionPoolSize, gpuId, gPool);
      Region<NDIM>*RegionLeft, *RegionRight;
      Result result;
      __syncthreads();

      if (threadIdx.x == 0) {
        Bounds *bL, *bR;
        Region<NDIM>* R = &sRegionPool[0];
        result.err = R->result.err;
        result.avg = R->result.avg;
        result.bisectdim = R->result.bisectdim;

        int bisectdim = result.bisectdim;

        RegionLeft = R;
        RegionRight = &sRegionPool[sRegionPoolSize];

        bL = &RegionLeft->bounds[bisectdim];
        bR = &RegionRight->bounds[bisectdim];

        RegionRight->div = ++RegionLeft->div;

        if (RegionRight->div > maxdiv)
          maxdiv = RegionRight->div;
        for (int dim = 0; dim < NDIM; ++dim) {
          RegionRight->bounds[dim].lower = RegionLeft->bounds[dim].lower;
          RegionRight->bounds[dim].upper = RegionLeft->bounds[dim].upper;
        }

        bL->upper = bR->lower = 0.5 * (bL->lower + bL->upper);
      }

      sRegionPoolSize++;
      nregions++;
      __syncthreads();
      SampleRegionBlock<IntegT, T, NDIM>(
        d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool);
      __syncthreads();
      SampleRegionBlock<IntegT, T, NDIM>(
        d_integrand, sRegionPoolSize - 1, constMem, FEVAL, NSETS, sRegionPool);
      __syncthreads();

      if (threadIdx.x == 0) {
        Result* rL = &RegionLeft->result;
        Result* rR = &RegionRight->result;

        T diff = rL->avg + rR->avg - result.avg;
        diff = fabs(.25 * diff);
        T err = rL->err + rR->err;

        if (err > 0) {
          T c = 1 + 2 * diff / err;
          rL->err *= c;
          rR->err *= c;
        }

        rL->err += diff;
        rR->err += diff;

        lasterr += rL->err + rR->err - result.err;
        lastavg += rL->avg + rR->avg - result.avg;

        if (Final == 0) {
          weightsum += w = 1 / fmax(lasterr * lasterr, ldexp(1., -104));
          avgsum += w * lastavg;
          sigsq = 1 / weightsum;
          avg = sigsq * avgsum;
        }

        ERR = Final ? lasterr : sqrt(sigsq);
        RESULT = Final ? lastavg : avg;

        if (batch.numRegions == 0)
          printf("%.20f +- %20f nregions:%i depth:%i\n",
                 RESULT,
                 ERR,
                 nregions,
                 RegionRight->div);
        /*
        if(numRegions!=0){
                atomicAdd(global_errorest+nregions, ERR);
                atomicAdd(numContributors+nregions, 1);
        }
        */

        /*
        if(abs(ERR/MaxErr(RESULT, epsrel,
        epsabs)-prev_ratio)< 1.5*required_ratio_decrease){ iterations_without++;
        }
        prev_ratio = abs(ERR/MaxErr(RESULT, epsrel, epsabs));
        */
      }
      __syncthreads();

      /*
     if(iterations_without >= 1 && numRegions!=0){
            break;
      }
      */

      /*
      if(numRegions!=0 && dPh1res[1]/numRegions > ERR){
              break;
      }
      */

      // if(ExistsIn(nregions, snap.sizes, snap.numSnapshots) == true)
      {

        /*if(threadIdx.x == 0){
                //printf("About to capture snapshot for %i th iteration at array
        index %i\n", snap.sizes[snapshot_id], current_snap_array_index);
                memcpy(&snap.arr[current_snap_array_index], gPool,
        sizeof(Region<NDIM>)*nregions); if(nregions == 4022 && threadIdx.x == 0
        && blockIdx.x == 0){ for(int n=0; n<nregions; n++){
                                printf("pre_insert,");
                                for(int i=0; i<NDIM; i++)
                                        printf("%f, %f, ",
        gPool[n].bounds[i].lower, gPool[n].bounds[i].upper); printf("%.17f,
        %.17f, %i\n", gPool[n].result.avg, gPool[n].result.err, gPool[n].div);
                        }
                }
        }*/

        //__syncthreads();
        // insert_global_store<T>(sRegionPool,
        // &snap.arr[current_snap_array_index], gpuId,
        // &snap.arr[current_snap_array_index], sRegionPoolSize);
        //__syncthreads();

        // if (threadIdx.x == 0)
        // gRegionPoolSize = gRegionPoolSize - (SM_REGION_POOL_SIZE / 2);
        //__syncthreads();

        /*
        if(threadIdx.x == 0){
                printf("Snapshot for iteration %i\n", nregions);
                Region<NDIM> *snapArr = &snap.arr[current_snap_array_index];
                for(int i=0; i < nregions; i++){
                        printf("region %i, %.20f, %.20f,", i,
        snapArr[i].result.avg, snapArr[i].result.err); for(int dim = 0; dim <
        NDIM; dim++){ printf("%.15f, %.15f,", snapArr[i].bounds[dim].upper,
        snapArr[i].bounds[dim].lower);
                        }
                        printf("%i, %i\n", snapArr[i].div, nregions);
                }
        }
        */

        // current_snap_array_index = current_snap_array_index +
        // snap.sizes[snapshot_id]; snapshot_id++;
      }
    }

    __syncthreads();
    // this is uncessary workload if we dont' care about collecting phase 2 data
    // for each block

    if (nregions != 1) {
      if ((sRegionPoolSize > 64 || nregions < 64) && batch.numRegions != 0) {
        INSERT_GLOBAL_STORE2<T>(
          sRegionPool, gPool, gpuId, gPool, sRegionPoolSize);
      }
      if ((sRegionPoolSize > 64 || nregions < 64) && batch.numRegions == 0) {
        insert_global_store<T>(
          sRegionPool, gPool, gpuId, gPool, sRegionPoolSize);
      }
    }

    __syncthreads();

    if (threadIdx.x == 0) {

      // if(nregions == 1)
      //	printf("%i, %.20f, %.20f, %.20f,%.20f, %i\n", blockIdx.x,
      // lastavg, lasterr, ERR,  fabs(lasterr-ERR), nregions);

      /*
               if(nregions!= 2048 && numRegions!=0){
                      for(int i=nregions+1; i<=2048; i++)
                              atomicAdd(global_errorest+i, ERR);
                              //printf("%i, %.20f, %.20f, %.20f,%.20f, %i\n",
         blockIdx.x, RESULT, ERR, 0,  0, i);
               }
      */
      int isActive = ERR > MaxErr(RESULT, epsrel, epsabs);

      if (ERR > (1e+10)) {
        RESULT = 0.0;
        ERR = 0.0;
        isActive = 1;
      }

      // if(nregions > 2048)
      //	 printf("[%i] nregions:%i\n", blockIdx.x, nregions);
      // printf("Final result%.20f \n %.20f\n", RESULT, ERR);
      // printf("%i, %i, %f, %.20f, %.20f, %.20f, %.20f\n", blockIdx.x,
      // nregions, ERR/MaxErr(RESULT, epsrel, epsabs),
      // batch.dRegionsIntegral[blockIdx.x], batch.dRegionsError[blockIdx.x],
      // RESULT, ERR);
      batch.activeRegions[blockIdx.x] = isActive;
      batch.dRegionsIntegral[blockIdx.x] = RESULT;
      batch.dRegionsError[blockIdx.x] = ERR;
      dRegionsNumRegion[blockIdx.x] = nregions;
    }
  }
}

#endif
