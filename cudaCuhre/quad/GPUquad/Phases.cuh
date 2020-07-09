#ifndef CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH
#define CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH

#include "../util/Volume.cuh"
#include "Sample.cuh"
#include <cooperative_groups.h>

#define FINAL 0

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

  template <typename IntegT, typename T, int NDIM>
  __device__ void
  ALIGN_GLOBAL_TO_SHARED(Region<NDIM> sRegionPool[], Region<NDIM>*& gPool)
  {

    // size_t intervalIndex = blockIdx.x;
    int idx = 0;
    int index = idx * BLOCK_SIZE + threadIdx.x;
    //---------------------------------------------
    // initializes shared memory with empty regions
    /*for (; idx < SM_REGION_POOL_SIZE / BLOCK_SIZE; ++idx) {

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
    }*/

    //-------------------------------------------------------
    // sets the bounds of the 1st region from 0 to 1 while the global bounds(1st
    // region's real boundaries) are assigned to sBound
    /*if (threadIdx.x == 0) {
      for (int dim = 0; dim < NDIM; ++dim) {

        sRegionPool[threadIdx.x].bounds[dim].lower = 0;
        sRegionPool[threadIdx.x].bounds[dim].upper = 1;
        T lower = dRegions[dim * numRegions + intervalIndex];
        sBound[dim].unScaledLower = lower;
        sBound[dim].unScaledUpper =
          lower + dRegionsLength[dim * numRegions + intervalIndex];
      }
    }*/
    //-------------------------------------------------------------

    //__syncthreads();

    // SampleRegionBlock<IntegT, T, NDIM>(
    //  d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool, lows, highs);

    /* if (threadIdx.x == 0) {
       gRegionPoolSize = (SM_REGION_POOL_SIZE / 2);
     }
         */

    __syncthreads();

    /*
            creates a deep copy of the regions in shared memory to global memory
            stores each shared memory region's corresponding index in global
       memory
    */
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
    // return 1;
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
                      Region<NDIM>*& gRegionPool,
                      int gpuId,
                      Region<NDIM>*& gPool)
  {

    /* if (threadIdx.x == 0 && GlobalMemCopy) {
            gPool = (Region<NDIM>*)malloc(
         sizeof(Region<NDIM>) * 2048);
     }*/
    __syncthreads();

    // Copy existing global regions into newly allocated spaced

    int iterationsPerThread = 0;
    /*if(GlobalMemCopy){

            for (iterationsPerThread = 0;
                     iterationsPerThread < gRegionPoolSize / BLOCK_SIZE;
                     ++iterationsPerThread) {
              size_t dataIndex = iterationsPerThread * BLOCK_SIZE + threadIdx.x;

              gPool[dataIndex] = gRegionPool[dataIndex];
              __syncthreads();
            }

            size_t dataIndex = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
            if (dataIndex < gRegionPoolSize) {
              gPool[dataIndex] = gRegionPool[dataIndex];
            }

    }*/

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

    // gRegionPool = gPool;
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

    // Comment 3 instructions these section if you are directly using new shared
    // memory instead of reusing shared memory

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
    for (offset = 0; (offset < MAX_GLOBALPOOL_SIZE / BLOCK_SIZE) &&
                     (offset < gRegionPoolSize / BLOCK_SIZE);
         offset++) {
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
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.err = -INFTY;
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.avg = 0;
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].div = 0;
    }

    index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      sRegionPool[index] = gPool[gRegionPos[index]];
      sRegionPool[(SM_REGION_POOL_SIZE / 2) + index].result.err = -INFTY;
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
    // If array  for regions in shared is full
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
                             T phase1_avgsum)
  {
    __shared__ Region<NDIM> sRegionPool[SM_REGION_POOL_SIZE];
    __shared__ Region<NDIM>* gPool;
    __shared__ T shighs[NDIM];
    __shared__ T slows[NDIM];
    __shared__ int max_global_pool_size;
	
	int maxdiv = 0;
	double smallest = 1;
	if(blockIdx.x == 1922 && threadIdx.x == 0)
		printf("inside phase 2\n");
    if (threadIdx.x == 0) {
      memcpy(slows, lows, sizeof(T) * NDIM);
      memcpy(shighs, highs, sizeof(T) * NDIM);
      max_global_pool_size = 2048;
      gPool = &ggRegionPool[blockIdx.x * MAX_GLOBALPOOL_SIZE];
    }

    InitSMemRegions(sRegionPool);
    // First compute sibling region
    // SET_FIRST_SHARED_MEM_REGION(sRegionPool, dRegions, dRegionsLength,
    // numRegions, GetSiblingIndex(blockIdx.x)); SampleRegionBlock<IntegT, T,
    // NDIM>(d_integrand, 0, &constMem, FEVAL, NSETS, sRegionPool, slows,
    // shighs);

    // T siblIntegral = sRegionPool[0].result.avg;
    // T siblError    = sRegionPool[0].result.err;

    int sRegionPoolSize = SET_FIRST_SHARED_MEM_REGION(
      sRegionPool, dRegions, dRegionsLength, numRegions, blockIdx.x);
    SampleRegionBlock<IntegT, T, NDIM>(
      d_integrand, 0, &constMem, FEVAL, NSETS, sRegionPool, slows, shighs);
    ALIGN_GLOBAL_TO_SHARED<IntegT, T, NDIM>(sRegionPool, gPool);

    ComputeErrResult<T, NDIM>(ERR, RESULT, sRegionPool);
	ERR = dRegionsError[blockIdx.x + numRegions];
  
    // TODO  : May be redundance sync
    __syncthreads();
    int nregions = sRegionPoolSize; // is only 1 at this point

    // prep for final = 0
    T lastavg = RESULT;
    T lasterr = ERR;
    T weightsum = 1 / fmax(ERR * ERR, ldexp(1., -104));
    T avgsum = weightsum * lastavg;

    T w = 0;
    T avg = 0;
    T sigsq = 0;
	
    /*
    if(threadIdx.x == 0)
            printf("%i, %.12f, %.12f, %i\n", blockIdx.x, RESULT, ERR,
    max_global_pool_size);
    */
	
	if(blockIdx.x == 1922 && threadIdx.x == 0)
		  printf("%i phase 2 regions %e vs %e\n", nregions, ERR, MaxErr(RESULT, epsrel, epsabs) );
	
    while (nregions < max_global_pool_size &&
           ERR > MaxErr(RESULT, epsrel, epsabs)) {

      sRegionPoolSize =
        EXTRACT_MAX<T, NDIM>(sRegionPool, gPool, sRegionPoolSize, gpuId, gPool);
      Region<NDIM>*RegionLeft, *RegionRight;
      Result result;

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
		
		if(blockIdx.x == 0 && threadIdx.x == 0 && nregions < 3){
			for(int dim = 0; dim < NDIM; dim++){
				printf("%f, %f -> %f, %f\n", RegionLeft->bounds[dim].lower, RegionLeft->bounds[dim].upper, 
											 sBound[dim].unScaledLower+RegionLeft->bounds[dim].lower*(sBound[dim].unScaledUpper - sBound[dim].unScaledLower),
											 sBound[dim].unScaledLower + RegionLeft->bounds[dim].upper*(sBound[dim].unScaledUpper - sBound[dim].unScaledLower));
			}
			printf("-----------------\n");
			for(int dim = 0; dim < NDIM; dim++){
				printf("%f, %f\n", sBound[dim].unScaledLower, sBound[dim].unScaledUpper);
			}
			printf("============================\n");
		}
		
		if(RegionRight->div > maxdiv)
			maxdiv = RegionRight->div;
		
        for (int dim = 0; dim < NDIM; ++dim) {
          RegionRight->bounds[dim].lower = RegionLeft->bounds[dim].lower;
          RegionRight->bounds[dim].upper = RegionLeft->bounds[dim].upper;
		  if(RegionLeft->bounds[dim].lower == RegionLeft->bounds[dim].upper)
			  printf("During computation Block %i, unscaled bounds:%e,%e div:%i\n", blockIdx.x, RegionLeft->bounds[dim].lower, RegionLeft->bounds[dim].upper, RegionRight->div);
		  //updates based on interval size
		  if(smallest > RegionLeft->bounds[dim].upper - RegionLeft->bounds[dim].lower)
			  smallest = RegionLeft->bounds[dim].upper - RegionLeft->bounds[dim].lower;
		  if(smallest > RegionRight->bounds[dim].upper - RegionRight->bounds[dim].lower)
			  smallest = RegionRight->bounds[dim].upper - RegionRight->bounds[dim].lower;
		  
		  /*if(smallest > RegionLeft->bounds[dim].upper)
			  smallest = RegionLeft->bounds[dim].upper;
		  if(smallest > RegionRight->bounds[dim].upper)
			  smallest = RegionRight->bounds[dim].upper;*/
        }
		
        bL->upper = bR->lower = 0.5 * (bL->lower + bL->upper);
		
      }
	
      sRegionPoolSize++;
      nregions++;
      __syncthreads();
      SampleRegionBlock<IntegT, T, NDIM>(
        d_integrand, 0, &constMem, FEVAL, NSETS, sRegionPool, slows, shighs);
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

        weightsum += w = 1 / fmax(lasterr * lasterr, ldexp(1., -104));
        avgsum += w * lastavg;
        sigsq = 1 / weightsum;
        avg = sigsq * avgsum;

        ERR = Final ? lasterr : sqrt(sigsq);
        RESULT = Final ? lastavg : avg;
      }
      __syncthreads();
	  
	  //this is uncessary workload if we dont' care about collecting phase 2 data for each block
	  
    }
	
	__syncthreads();
	if(sRegionPoolSize > 64 || nregions<64)
		INSERT_GLOBAL_STORE<T>(sRegionPool, gPool, gpuId, gPool);
	 __syncthreads();
	
    if (threadIdx.x == 0) {
      int isActive = ERR > MaxErr(RESULT, epsrel, epsabs);

      if (ERR > (1e+10)) {
        RESULT = 0.0;
        ERR = 0.0;
        isActive = 1;
      }
	  //printf("%.30f, %i, %e, %e, %i\n", smallest, maxdiv, ERR, RESULT, isActive);
      /*if(blockIdx.x == 0){
              printf("===============\n");
              printf("Phase 1 brought stats:%.12f +- %.12f || Sums: (%f,%f)\n",
      phase1_lastavg, phase1_lasterr, phase1_avgsum, phase1_weightsum);
              printf("Phase 2 BL0 Local contribution %.12f +-%.12f Sums:%f\n",
      lastavg, lasterr, avgsum, weightsum); printf("===============\n");
      }*/

      activeRegions[blockIdx.x] = isActive;
      // dRegionsIntegral[blockIdx.x] 	= RESULT;
      // dRegionsError[blockIdx.x] 	= ERR;
      dRegionsIntegral[blockIdx.x] = RESULT;
      dRegionsError[blockIdx.x] = ERR;
      dRegionsNumRegion[blockIdx.x] = nregions;
	  if(blockIdx.x == 1922)
		  printf("%i phase 2 regions\n", nregions);
      // free(gPool);
    }
  }
}

#endif
