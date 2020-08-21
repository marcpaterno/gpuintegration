#ifndef CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH
#define CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH

#include "../util/Volume.cuh"
#include "Sample.cuh"
#include <cooperative_groups.h>

#define FINAL 0
//added for variadic cuprintf
#include <stdio.h>
#include <stdarg.h>
namespace quad {

  /*__device__ __host__
  double
  Sq(double x){return x*x;}*/

  /* struct weightsum_functor {
       __device__ __host__
       double
       operator()(double err){return  1/fmax(Sq(err), ldexp(1., -104));}
   };

 template<typename T>
 double
 ComputeWeightSum(T *errors, size_t size){
        thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(errors);
        thrust::transform(d_ptr, d_ptr + size, d_ptr,  weightsum_functor());
        double weightsum = thrust::reduce(d_ptr, d_ptr + size);
        double sigsq = 1/weightsum;
 }	*/

  /*template<typename T>
  void ApplyFinal0(T &avg, T &err, T &weightsum, T &avgsum, T &chisum,  T
&chisqsum, T &chisq, const T guess){ double w = 0, sigsq = 0; weightsum += w =
1/Max(sqrt(err), ldexp(1., -104)); sigsq = 1/weightsum; avgsum += w*avg; avg =
sigsq*avgsum; chisum += w *= avg - guess; chisqsum += w*avg; chisq = chisqsum -
avg*chisum;
}*/

  /*template<typename T>
double
ComputeWeightSum(T *errors, size_t size){
   thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(errors);
   thrust::transform(d_ptr, d_ptr + size, d_ptr,  weightsum_functor());
   double weightsum = thrust::reduce(d_ptr, d_ptr + size);
   double sigsq = 1/weightsum;
}*/

__device__ void cuprintf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
	
	if(threadIdx.x == 0){
		while (*fmt != '\0') {
			if (*fmt == 'd') {
				int i = va_arg(args, int);
				printf("%d\n", i);
			} 
			else if (*fmt == 'c') {
				int c = va_arg(args, int);
				printf("%c\n", c);
			} 
			else if (*fmt == 'f') {
				double d = va_arg(args, double);
				printf("%f\n", d);
			}
			++fmt;
		}
	}
    va_end(args);
}


  template <typename IntegT, typename T, int NDIM>
  __device__ void
  INIT_REGION_POOL(IntegT* d_integrand,
                   T* dRegions,
                   T* dRegionsLength,
                   size_t numRegions,
                   Structures<T>* constMem,
                   int FEVAL,
                   int NSETS,
                   Region<NDIM> sRegionPool[],
                   T* lows,
                   T* highs)
  {

    size_t index = blockIdx.x;

    if (threadIdx.x == 0) {
      for (int dim = 0; dim < NDIM; ++dim) {
        T lower = dRegions[dim * numRegions + index];

        sRegionPool[threadIdx.x].bounds[dim].lower = 0;
        sRegionPool[threadIdx.x].bounds[dim].upper = 1;

        sBound[dim].unScaledLower = lower;
        sBound[dim].unScaledUpper =
          lower + dRegionsLength[dim * numRegions + index];
        sRegionPool[threadIdx.x].div = 0;
      }
    }

    __syncthreads();
    SampleRegionBlock<IntegT, T, NDIM>(
      d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool, lows, highs);
    __syncthreads();
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
              int numRegions,
              T epsrel,
              T epsabs)
  {

    if (threadIdx.x == 0 && blockIdx.x < numRegions) {
      int fail = 0;

      T selfErr = dRegionsError[blockIdx.x + numRegions];
      T selfRes = dRegionsIntegral[blockIdx.x + numRegions];

      // that's how indices to the right to find the sibling
      // but we want the sibling to be found at the second half of the array
      // only, to avoid race conditions

      int siblingIndex = (numRegions / 2) + blockIdx.x;
      if (siblingIndex < numRegions) {
        siblingIndex += numRegions;
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

      if ((selfErr / MaxErr(selfRes, epsrel, epsabs)) > 1) {
        fail = 1;
        newErrs[blockIdx.x] = 0;
        dRegionsIntegral[blockIdx.x] = 0;
      } else {
        newErrs[blockIdx.x] = selfErr;
      }

      activeRegions[blockIdx.x] = fail;
      newErrs[blockIdx.x + numRegions] = selfErr;
    }
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
                       T* highs)
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
    int fail = 0;

    INIT_REGION_POOL<IntegT>(d_integrand,
                             dRegions,
                             dRegionsLength,
                             numRegions,
                             &constMem,
                             FEVAL,
                             NSETS,
                             sRegionPool,
                             slows,
                             shighs);

    if (threadIdx.x == 0) {
      ERR = sRegionPool[threadIdx.x].result.err;
      RESULT = sRegionPool[threadIdx.x].result.avg;
      T ratio = ERR / MaxErr(RESULT, epsrel, epsabs);
      int fourthDiffDim = sRegionPool[threadIdx.x].result.bisectdim;
      dRegionsIntegral[gridDim.x + blockIdx.x] = RESULT;
      dRegionsError[gridDim.x + blockIdx.x] = ERR;

      if (ratio > 1) {
        fail = 1;
      }

      activeRegions[blockIdx.x] = fail;
      subDividingDimension[blockIdx.x] = fourthDiffDim;
      dRegionsIntegral[blockIdx.x] = RESULT;
      dRegionsError[blockIdx.x] = ERR;

      __syncthreads();

      if (ratio > 1 && numRegions == 1) {
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
                              T* dRegions,
                              T* dRegionsLength,
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
        sBound[dim].unScaledLower = lower;
        sBound[dim].unScaledUpper =
          lower + dRegionsLength[dim * numRegions + intervalIndex];
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
    //*lows & highs are user defined bounds, they are ok being stored in sBound
    size_t intervalIndex = blockIndex;
    if (threadIdx.x == 0) {
      gRegionPoolSize = (SM_REGION_POOL_SIZE / 2);
      for (int dim = 0; dim < NDIM; ++dim) {
        sRegionPool[threadIdx.x].bounds[dim].lower = 0;
        sRegionPool[threadIdx.x].bounds[dim].upper = 1;
        T lower = lows[dim * numRegions + intervalIndex];
        sBound[dim].unScaledLower = lower;
        sBound[dim].unScaledUpper = highs[dim * numRegions + intervalIndex];
      }
    }
    __syncthreads();
    return 1;
  }
	
  template <typename T, int NDIM>
  __device__ int
  set_first_shared_mem_region(Region<NDIM> sRegionPool[],
                              Region<NDIM>* ggRegionPool,
                              size_t numRegions,
                              size_t blockIndex)
  {

    size_t intervalIndex = blockIndex;
    if (threadIdx.x == 0) {
      gRegionPoolSize = (SM_REGION_POOL_SIZE / 2);

      for (int dim = 0; dim < NDIM; ++dim) {
        sRegionPool[threadIdx.x].bounds[dim].lower = 0;
        sRegionPool[threadIdx.x].bounds[dim].upper = 1;
        T lower = ggRegionPool[intervalIndex].bounds[dim].lower;
        sBound[dim].unScaledLower = lower;
        sBound[dim].unScaledUpper = ggRegionPool[intervalIndex].bounds[dim].upper;
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
  INSERT_GLOBAL_STORE(Region<NDIM>* sRegionPool,
                      Region<NDIM>* gRegionPool,
                      int gpuId,
                      Region<NDIM>* gPool)
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

    for (iterationsPerThread = 0; iterationsPerThread < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE; ++iterationsPerThread) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      gPool[gRegionPos[index]] = sRegionPool[index];
	  //printf("%i->%lu (%.20f)\n", index, gRegionPos[index], sRegionPool[index].result.err);
      if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize){
        gPool[gRegionPoolSize + index] =  sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
        //printf("%i->%lu (%.20f)\n", gRegionPoolSize + index, (SM_REGION_POOL_SIZE / 2) + index, sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.err);
	  }
		
      for (int dim = 0; dim < NDIM; dim++) {
        gPool[gRegionPos[index]].bounds[dim].lower =  sBound[dim].unScaledLower +  sRegionPool[index].bounds[dim].lower * (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
        gPool[gRegionPos[index]].bounds[dim].upper =  sBound[dim].unScaledLower +  sRegionPool[index].bounds[dim].upper * (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);

        if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize) {
          gPool[gRegionPoolSize + index].bounds[dim].lower =  
			sBound[dim].unScaledLower + sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].bounds[dim].lower *(sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
          gPool[gRegionPoolSize + index].bounds[dim].upper =
            sBound[dim].unScaledLower + sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].bounds[dim].upper *(sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
        }
      }
    }

    __syncthreads();
    int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      gPool[gRegionPos[index]] = sRegionPool[index];
	  //printf("%i->%lu (%.20f)\n", index, gRegionPos[index], sRegionPool[index].result.err);
      if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize){
        gPool[gRegionPoolSize + index] =
          sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
		//printf("%i->%lu (%.20f)\n", gRegionPoolSize + index, (SM_REGION_POOL_SIZE / 2) + index, sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.err);
	  }
	  
      for (int dim = 0; dim < NDIM; dim++) {
        gPool[gRegionPos[index]].bounds[dim].lower = sBound[dim].unScaledLower + sRegionPool[index].bounds[dim].lower * (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
        gPool[gRegionPos[index]].bounds[dim].upper = sBound[dim].unScaledLower + sRegionPool[index].bounds[dim].upper *(sBound[dim].unScaledUpper - sBound[dim].unScaledLower);

        if ((SM_REGION_POOL_SIZE / 2) + index < sRegionPoolSize) {
          gPool[gRegionPoolSize + index].bounds[dim].lower = sBound[dim].unScaledLower + sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].bounds[dim].lower * (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
          gPool[gRegionPoolSize + index].bounds[dim].upper = sBound[dim].unScaledLower + sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].bounds[dim].upper * (sBound[dim].unScaledUpper - sBound[dim].unScaledLower);
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
	//offset >>=1 offset /= 2
    for (size_t offset = gSize / 2; offset > 0; offset >>= 1) {
      int idx = 0;
	  
	  //main loop
      for (idx = 0; idx < offset / BLOCK_SIZE; ++idx) {
        size_t index = idx * BLOCK_SIZE + threadIdx.x;
        if (index < offset) {
          if (serror[index] < serror[index + offset]) {
            swap(serror[index], 	serror[index + offset]);
            swap(serrorPos[index], 	serrorPos[index + offset]);
          }
        }
      }
	  
	  //leftovers from loop above
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
	
    // Comment 3 instructions these section if you are directly using new shared
    // memory instead of reusing shared memory
	
    T* sarray = (T*)&sRegionPool[0];
	
    if (threadIdx.x == 0) {
	  //if double the gRegionSize < SM_REGION_POOL_SIZE we can sort in shared memory
      if ((gRegionPoolSize * sizeof(T) + gRegionPoolSize * sizeof(size_t)) < sizeof(Region<NDIM>) * SM_REGION_POOL_SIZE) {
        serror = &sarray[0];
        // TODO:Size of sRegionPool vs sarray constrain
        serrorPos = (size_t*)&sarray[gRegionPoolSize];
      } 
	  else {
		  //otherwise we need to allocate more space for the sorting
        serror = (T*)malloc(sizeof(T) * gRegionPoolSize);
        serrorPos = (size_t*)malloc(sizeof(size_t) * gRegionPoolSize);
      }
    }
    __syncthreads();
	
	//TODO:why use MAX_GLOBALPOOL_SIZE, could this cause the zeroes in the arrays
    int offset = 0;
	//TODO: CHECK EFFECT ON PERFORMANCE IF ANY
    //for (offset = 0; (offset < MAX_GLOBALPOOL_SIZE / BLOCK_SIZE) && (offset < gRegionPoolSize / BLOCK_SIZE); offset++) {
	for (offset = 0; offset < gRegionPoolSize / BLOCK_SIZE; offset++) {
      size_t regionIndex 	 = offset * BLOCK_SIZE + threadIdx.x;
      serror[regionIndex] 	 = gPool[regionIndex].result.err;
      serrorPos[regionIndex] = regionIndex;
    }
	
    size_t regionIndex = offset * BLOCK_SIZE + threadIdx.x;
    if (regionIndex < gRegionPoolSize) {
      serror[regionIndex] 	 = gPool[regionIndex].result.err;
      serrorPos[regionIndex] = regionIndex;
    }
	
    __syncthreads();
    for (int k = 0; k < (SM_REGION_POOL_SIZE / 2); ++k) {
      EXTRACT_MAX<T>(&serror[k], &serrorPos[k], gRegionPoolSize - k);
	  //if(gRegionPoolSize == 3968 && threadIdx.x == 0)
	//	printf("serror[%i] estimated %.20f\n", k, serror[k]);
    }
	
	//for(int k = SM_REGION_POOL_SIZE/2; k< gRegionPoolSize; k++){
	//	if(threadIdx.x == 0 && gRegionPoolSize == 3968)
			//printf(" serror[%i] estimated %.20f\n", k, serror[k]);
	//}
	
    int iterationsPerThread  = 0;
    for (iterationsPerThread = 0; iterationsPerThread < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE; ++iterationsPerThread) {
      int index  		= iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      size_t pos 		= serrorPos[index];
      gRegionPos[index] = pos;
    }
	
    int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      size_t pos    	= serrorPos[index];
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
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.avg =0;
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
    // If array  for regions in shared is full, clear it and update the global memory array
    if (sSize == SM_REGION_POOL_SIZE) {

      INSERT_GLOBAL_STORE<T>(sRegionPool, gRegionPool, gpuId, gPool);
      __syncthreads();
	
      // gRegionPool = gPool;
      EXTRACT_TOPK<T>(sRegionPool, gRegionPool, gPool);
      sSize = (SM_REGION_POOL_SIZE / 2);
      __syncthreads();
    }
	
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
  __device__ void SetBlockVars(T* slows, T* shighs, int &max_global_pool_size, Region<NDIM>*& gPool, const T* lows, const T* highs, const int max_regions, /*const*/ Region<NDIM>* ggRegionPool){
	if (threadIdx.x == 0) {
      memcpy(slows, lows, sizeof(T) * NDIM);
      memcpy(shighs, highs, sizeof(T) * NDIM);
      max_global_pool_size = max_regions;
      gPool = &ggRegionPool[blockIdx.x * max_regions];
    }  
	 __syncthreads(); 
  }

  template<typename T>
  __device__
  bool ExistsIn(T val, T* array, int array_size){
	  for(int i=0; i< array_size; i++){
		  if(array[i] == val)
			  return true;
	  }
	 return false;
  }
  
  template <typename IntegT, typename T, int NDIM>
  __global__ void
  BLOCK_INTEGRATE_GPU_PHASE2(IntegT* d_integrand,
                             T* dRegions,
                             T* dRegionsLength,
                             size_t numRegions,
                             T* dRegionsIntegral,
                             T* dRegionsError,
                             int* dRegionsNumRegion,
                             int* activeRegions,
                             int* subDividingDimension,
                             T epsrel,
                             T epsabs,
                             int gpuId,
                             Structures<T> constMem,
                             int FEVAL,
                             int NSETS,
                             double* exitCondition,
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
                             int phase1_type = 0,
                             Region<NDIM>* phase1_regs = nullptr,
							 Snapshot<NDIM> snap = Snapshot<NDIM>())
  {
    __shared__ Region<NDIM> sRegionPool[SM_REGION_POOL_SIZE];
    __shared__ Region<NDIM>* gPool;
    __shared__ T shighs[NDIM];
    __shared__ T slows[NDIM];
    __shared__ int max_global_pool_size;
	__shared__ int iterations_without;
	T pseudo_global_val;
	T pseudo_global_err;
	T prev_ratio = 0;
	T prev_error = 0;
	int maxdiv 		= 0;
	//T origerr;
	//T origavg;
    int current_snap_array_index = 0;
	
	//SetBlockVars<T, NDIM>(slows, shighs, max_global_pool_size, gPool, lows, highs, max_regions, ggRegionPool);
	double personal_estimate_ratio = 0;
	int local_region_cap = 32734;
	
    if (threadIdx.x == 0) {
      memcpy(slows, lows, sizeof(T) * NDIM);
      memcpy(shighs, highs, sizeof(T) * NDIM);
      max_global_pool_size = max_regions;
      gPool = &ggRegionPool[blockIdx.x * max_regions];
	  iterations_without = 0;
    }
	
	pseudo_global_val = dPh1res[0];
	pseudo_global_err = dPh1res[1];
	
    __syncthreads();              // added for testing 
    InitSMemRegions(sRegionPool); // sets every region in shared memory to zero
    int sRegionPoolSize = 1;
    __syncthreads();
	//temporary solution
    if (phase1_type == 0)
      SET_FIRST_SHARED_MEM_REGION(sRegionPool, dRegions, dRegionsLength, numRegions, blockIdx.x);
    else if (numRegions == 0)
      set_first_shared_mem_region(sRegionPool, dRegions, dRegionsLength, numRegions, blockIdx.x);
    else 
      set_first_shared_mem_region<T, NDIM>(sRegionPool, phase1_regs, numRegions, blockIdx.x);
    
    __syncthreads();
	
    SampleRegionBlock<IntegT, T, NDIM>(d_integrand, 0, &constMem, FEVAL, NSETS, sRegionPool, slows, shighs);
    ALIGN_GLOBAL_TO_SHARED<IntegT, T, NDIM>(sRegionPool, gPool);
    ComputeErrResult<T, NDIM>(ERR, RESULT, sRegionPool);
	
	//temporary solution
    if (threadIdx.x == 0 && phase1_type == 0) {
       ERR = dRegionsError[blockIdx.x + numRegions]; 
    } else {
      if (threadIdx.x == 0 && numRegions != 0) {
        ERR = phase1_regs[blockIdx.x].result.err;
      }
    }
	
    __syncthreads();
	prev_error = ERR;
	prev_ratio = ERR/MaxErr(RESULT, epsrel, epsabs);
	
	personal_estimate_ratio = RESULT/dPh1res[0];
	
	if(numRegions != 0)
		local_region_cap = min(2048, (int)(personal_estimate_ratio*2048*numRegions));
	
	T required_ratio_decrease = abs(1 - prev_ratio)/local_region_cap;
	
    int nregions = sRegionPoolSize; // is only 1 at this point
    T  lastavg = RESULT;
    T  lasterr = ERR;
    T  weightsum = 1 / fmax(ERR * ERR, ldexp(1., -104));
    T  avgsum = weightsum * lastavg;
	
    T w = 0;
    T avg = 0;
    T sigsq = 0;
	int snapshot_id = 0;
	
    while (nregions < max_global_pool_size &&
           (ERR > MaxErr(RESULT, epsrel, epsabs)) && nregions < local_region_cap) {
	  
      sRegionPoolSize = EXTRACT_MAX<T, NDIM>(sRegionPool, gPool, sRegionPoolSize, gpuId, gPool);
      Region<NDIM>*RegionLeft, *RegionRight;
      Result result;
	  if(nregions == 3968 && threadIdx.x == 0 && blockIdx.x == 0){
				/*for(int n=0; n<SM_REGION_POOL_SIZE; n++){
					printf("shared,");
					for(int i=0; i<NDIM; i++)
						printf("%f, %f, ", sRegionPool[n].bounds[i].lower, sRegionPool[n].bounds[i].upper);
					printf("%.17f, %.17f, %i\n", sRegionPool[n].result.avg, sRegionPool[n].result.err, sRegionPool[n].div);
				}*/
				/*for(int i=0; i < nregions; i++){
					
					for(int dim = 0; dim < NDIM; dim++){
						printf("%.15f, %.15f,", gPool[i].bounds[dim].upper, gPool[i].bounds[dim].lower);
					}
					printf("%.20f, %.20f,", gPool[i].result.avg, gPool[i].result.err);
					printf("%i\n", gPool[i].div);
				}*/	
	  }
		  /*for(int n=0; n<nregions; n++){
				for(int i=0; i<NDIM; i++)
					printf("%f, %f, ", gPool[n].bounds[i].lower, gPool[n].bounds[i].upper);
				printf("%.17f, %.17f, %i\n", gPool[n].result.avg, gPool[n].result.err, gPool[n].result.bisectdim);
			}*/
	  __syncthreads();
	  //needed for investingating dcuhre differences
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
		  /*for debugging, will essentially print each selected region
		   if(blockIdx.x == 0 && numRegions == 0)
				printf("%f, %f, ", RegionLeft->bounds[dim].lower, RegionLeft->bounds[dim].upper);
		  */
        }
		
		//printf("%i, %i, ", bisectdim, RegionLeft->div-1);
        bL->upper = bR->lower = 0.5 * (bL->lower + bL->upper);
      }
		
      sRegionPoolSize++;
      nregions++;
      __syncthreads();
      SampleRegionBlock<IntegT, T, NDIM>(d_integrand, 0, &constMem, FEVAL, NSETS, sRegionPool, slows, shighs);
	  __syncthreads();
      SampleRegionBlock<IntegT, T, NDIM>(d_integrand,
                                         sRegionPoolSize - 1,
                                         &constMem,
                                         FEVAL,
                                         NSETS,
                                         sRegionPool,
                                         slows,
                                         shighs);
      __syncthreads();
		
      if (threadIdx.x == 0) {
        Result* rL = &RegionLeft->result;
        Result* rR = &RegionRight->result;
		
		/* 
		 printf("Parent:%a +- %a \n", result.avg, result.err); 
		 printf("Pre Ref L: %a, %a diff:%a\n", rL->avg, rL->err, rL->avg + rR->avg - result.avg); 
		 printf("Pre Ref R: %a, %a\n", rR->avg, rR->err); 
		 printf("Parent:%.15f +- %.15f \n", result.avg, result.err); 
		 printf("Pre Ref L: %.15f, %.15f diff:%.15f\n", rL->avg, rL->err, rL->avg + rR->avg - result.avg); 
		 printf("Pre Ref R: %a, %a\n", rR->avg, rR->err); 
		 printf("lastavg:%.15f, lasterr:%.15f, weightsum:%15f, avgsum:%.15f\n", lastavg, lasterr , weightsum, avgsum);
		 */
		
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
		
		/*
		printf("After Ref L: %.15f, %.15f diff:%.15f\n", rL->avg, rL->err,  diff); 
		printf("After Ref R: %.15f, %.15f nregions:%i\n", rR->avg, rR->err, diff, nregions);
        printf("After Ref L: %a, %a diff:%a\n", rL->avg, rL->err,  diff); 
		printf("After Ref R: %a, %a\n", rR->avg, rR->err, diff);*/
		
        lasterr += rL->err + rR->err - result.err;
        lastavg += rL->avg + rR->avg - result.avg;
		
        weightsum 	+= w = 1 / fmax(lasterr * lasterr, ldexp(1., -104));
        avgsum 		+= w * lastavg;
        sigsq = 1 / weightsum;
        avg = sigsq * avgsum;
		
		pseudo_global_err = pseudo_global_err + lasterr - ERR;
		pseudo_global_val = pseudo_global_val + lastavg - RESULT;
		
        ERR 	= Final ? lasterr : sqrt(sigsq);
        RESULT 	= Final ? lastavg : avg;
		
		
		
		if(abs(ERR/MaxErr(RESULT, epsrel, epsabs)-prev_ratio)< required_ratio_decrease){
			iterations_without++;
		}
		prev_ratio = abs(ERR/MaxErr(RESULT, epsrel, epsabs));
		
		
		
		/*if(blockIdx.x == 6)
			printf("%i, %.20f, %.20f, local ratio:%.20f pseudo_global ratio:%.20f local cap:%i rrd:%10f ard:%10f\n", threadIdx.x, 
		RESULT, 
		ERR,
		ERR/MaxErr(RESULT, epsrel, epsabs),
		pseudo_global_err/MaxErr(pseudo_global_val, epsrel, epsabs),
		local_region_cap,
		required_ratio_decrease,
		abs(ERR/MaxErr(RESULT, epsrel, epsabs)-prev_ratio));*/
		
		//printf("iteration %i gRegionPoolSize:%lu\n", nregions, gRegionPoolSize);
		//if(threadIdx.x == 0)
		//printf("%.20f, %.20f, %.20f, %.20f, %.20f, %.20f, %i\n", rL->avg, rL->err, rR->avg, rR->err, RESULT, ERR, nregions);
	  }
      __syncthreads();
	  
	  if(iterations_without >= 1 && numRegions!=0){
		break;
	  }
	  if(numRegions!=0 && pseudo_global_err < MaxErr(pseudo_global_val, epsrel, epsabs)){
		  if(threadIdx.x == 0)
			printf("%i, Pseudo Exit ratio: %f Local Ratio:%f\n", blockIdx.x, 
																pseudo_global_err/MaxErr(pseudo_global_val, epsrel, epsabs), 
																ERR/MaxErr(RESULT, epsrel, epsabs));
		  break;
	  }
	  
	  /*if(numRegions!=0 && dPh1res[1]/numRegions > ERR){
		  if(threadIdx.x == 0)
			  printf("%i Exiting because of budget error nregions:%i\n", blockIdx.x, nregions);
	  }*/
	  
	  if(0 == 1 && ExistsIn(nregions, snap.sizes, snap.numSnapshots) == true){
			
			/*if(threadIdx.x == 0){
				//printf("About to capture snapshot for %i th iteration at array index %i\n", snap.sizes[snapshot_id], current_snap_array_index);
				memcpy(&snap.arr[current_snap_array_index], gPool, sizeof(Region<NDIM>)*nregions);
				if(nregions == 4022 && threadIdx.x == 0 && blockIdx.x == 0){
					for(int n=0; n<nregions; n++){
						printf("pre_insert,");
						for(int i=0; i<NDIM; i++)
							printf("%f, %f, ", gPool[n].bounds[i].lower, gPool[n].bounds[i].upper);
						printf("%.17f, %.17f, %i\n", gPool[n].result.avg, gPool[n].result.err, gPool[n].div);
					}
				}
			}*/
			
			__syncthreads();
			insert_global_store<T>(sRegionPool, &snap.arr[current_snap_array_index], gpuId, &snap.arr[current_snap_array_index], sRegionPoolSize);
			__syncthreads();
			
			if (threadIdx.x == 0)
				gRegionPoolSize = gRegionPoolSize - (SM_REGION_POOL_SIZE / 2);
			__syncthreads();
			
			if(nregions == 4022 && threadIdx.x == 0 && blockIdx.x == 0){
				/*for(int n=0; n<nregions; n++){
					printf("gpool,");
					for(int i=0; i<NDIM; i++)
						printf("%f, %f, ", gPool[n].bounds[i].lower, gPool[n].bounds[i].upper);
					printf("%.17f, %.17f, %i\n", gPool[n].result.avg, gPool[n].result.err, gPool[n].div);
				}*/
				
				/*for(int n=0; n < SM_REGION_POOL_SIZE; n++){
					printf("shared,");
					for(int i=0; i<NDIM; i++)
						printf("%f, %f, ", sRegionPool[n].bounds[i].lower, sRegionPool[n].bounds[i].upper);
					printf("%.17f, %.17f, %i\n", sRegionPool[n].result.avg, sRegionPool[n].result.err, sRegionPool[n].div);
				}*/
			}
			
			/*
			if(threadIdx.x == 0){
				printf("Snapshot for iteration %i\n", nregions);
				Region<NDIM> *snapArr = &snap.arr[current_snap_array_index];
				for(int i=0; i < nregions; i++){
					printf("region %i, %.20f, %.20f,", i, snapArr[i].result.avg, snapArr[i].result.err);
					for(int dim = 0; dim < NDIM; dim++){
						printf("%.15f, %.15f,", snapArr[i].bounds[dim].upper, snapArr[i].bounds[dim].lower);
					}
					printf("%i, %i\n", snapArr[i].div, nregions);
				}	
			}
			*/
			
			current_snap_array_index = current_snap_array_index + snap.sizes[snapshot_id];
			snapshot_id++;
		}
    }
	
    __syncthreads();
    // this is uncessary workload if we dont' care about collecting phase 2 data
    // for each block
    if (nregions != 1) {
      if ((sRegionPoolSize > 64 || nregions < 64) && numRegions != 0) {
        INSERT_GLOBAL_STORE2<T>(sRegionPool, gPool, gpuId, gPool, sRegionPoolSize);
      }
      if ((sRegionPoolSize > 64 || nregions < 64) && numRegions == 0) {
        insert_global_store<T>(sRegionPool, gPool, gpuId, gPool, sRegionPoolSize);
      }
    }
	
    __syncthreads();
	
    if (threadIdx.x == 0) {
      int isActive = ERR > MaxErr(RESULT, epsrel, epsabs);
	
      if (ERR > (1e+10)) {
        RESULT 		= 0.0;
        ERR 		= 0.0;
        isActive 	= 1;
      }
	  
	 // printf("Final result%.20f \n %.20f\n", RESULT, ERR);
      activeRegions[blockIdx.x] 	= isActive;
      dRegionsIntegral[blockIdx.x] 	= RESULT;
      dRegionsError[blockIdx.x] 	= ERR;
      dRegionsNumRegion[blockIdx.x] = nregions;
    }
  }
}

#endif
