#ifndef CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH
#define CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH

#include "GPUQuadSample.cuh"
#include <cooperative_groups.h>

namespace quad {

  template <typename IntegT, typename T, int NDIM>
  __device__ void
  INIT_REGION_POOL(IntegT* d_integrand,
                   T* dRegions,
                   T* dRegionsLength,
                   size_t numRegions,
                   Structures<T>* constMem,
                   int FEVAL,
                   int NSETS, 
				   Region<NDIM> sRegionPool[])
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
    SampleRegionBlock<IntegT, T, NDIM>(d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool);
    __syncthreads();
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
      if (siblingIndex < numRegions){
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
  INTEGRATE_GPU_PHASE1( IntegT* d_integrand,
                        T* dRegions,
                        T* dRegionsLength,
                        size_t numRegions,
                        T* dRegionsIntegral,
                        T* dRegionsError,
                        T* dParentsIntegral,
                        T* dParentsError,
                        int* activeRegions,
                        int* subDividingDimension,
                        T epsrel,
                        T epsabs,
                        Structures<T> constMem,
                        int FEVAL,
                        int NSETS)
  {
	__shared__ Region<NDIM> sRegionPool[SM_REGION_POOL_SIZE];
	
    T ERR = 0, RESULT = 0;
    int fail = 0;

    INIT_REGION_POOL<IntegT>(
      d_integrand, dRegions, dRegionsLength, numRegions, &constMem, FEVAL, NSETS, sRegionPool);

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

  template <typename IntegT, typename T, int NDIM>
  __device__ int
  INIT_REGION_POOL(IntegT* d_integrand,
                   T* dRegions,
                   T* dRegionsLength,
                   int* subDividingDimension,
                   size_t numRegions,
                   Structures<T>* constMem,
                   int FEVAL,
                   int NSETS,
				   Region<NDIM> sRegionPool[],
				   Region<NDIM>*& gPool)
  {

    size_t intervalIndex = blockIdx.x;
    int idx = 0;

    //SM_REGION_POOL_SIZE = 128 (quad.h) BLOCK_SIZE=256
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

    if (threadIdx.x == 0) {
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

    SampleRegionBlock<IntegT, T, NDIM>(d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool);

    if (threadIdx.x == 0) {
      gPool = (Region<NDIM>*)malloc(sizeof(Region<NDIM>) * (SM_REGION_POOL_SIZE / 2));
	  if(gPool == nullptr)
		  printf("Block %i failed to malloc gPool in Phase 2 Init_Region_Pool\n", blockIdx.x);
      gRegionPoolSize = (SM_REGION_POOL_SIZE / 2); 
    }

    __syncthreads();

    for (idx = 0; idx < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE; ++idx) {
      size_t index = idx * BLOCK_SIZE + threadIdx.x;
      gRegionPos[index] = index;
      gPool[index] = sRegionPool[index];
    }

    index = idx * BLOCK_SIZE + threadIdx.x;
    if (index < (SM_REGION_POOL_SIZE / 2)) {
      gRegionPos[index] = index;
      gPool[index] 		= sRegionPool[index];
    }
    return 1;
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
  INSERT_GLOBAL_STORE(Region<NDIM>* sRegionPool, Region<NDIM>* gRegionPool, int gpuId, Region<NDIM>*& gPool)
  {

    if (threadIdx.x == 0) {
      gPool = (Region<NDIM>*)malloc(sizeof(Region<NDIM>) *
                              (gRegionPoolSize + (SM_REGION_POOL_SIZE / 2)));
    }
    __syncthreads();

    // Copy existing global regions into newly allocated spaced
    int iterationsPerThread = 0;
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

    for (iterationsPerThread = 0;
         iterationsPerThread < (SM_REGION_POOL_SIZE / 2) / BLOCK_SIZE;
         ++iterationsPerThread) {
      int index = iterationsPerThread * BLOCK_SIZE + threadIdx.x;
      gPool[gRegionPos[index]] = sRegionPool[index];
      gPool[gRegionPoolSize + index] =
        sRegionPool[(SM_REGION_POOL_SIZE / 2) + index];
    }

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
      free(gRegionPool);
    }
    __syncthreads();

    gRegionPool = gPool;
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
  EXTRACT_TOPK(Region<NDIM>* sRegionPool, Region<NDIM>* gRegionPool, Region<NDIM>* gPool)
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
      serror[regionIndex] = gRegionPool[regionIndex].result.err;
      serrorPos[regionIndex] = regionIndex;
    }
    size_t regionIndex = offset * BLOCK_SIZE + threadIdx.x;
    if (regionIndex < gRegionPoolSize) {
      serror[regionIndex] = gRegionPool[regionIndex].result.err;
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
  EXTRACT_MAX(Region<NDIM>* sRegionPool, Region<NDIM>* gRegionPool, size_t sSize, int gpuId, Region<NDIM>*& gPool)
  {
    // If array  for regions in shared is full
    if (sSize == SM_REGION_POOL_SIZE) {

      INSERT_GLOBAL_STORE<T>(sRegionPool, gRegionPool, gpuId, gPool);
      __syncthreads();

      gRegionPool = gPool;
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
                             double* exitCondition)
  {
	__shared__ Region<NDIM> sRegionPool[SM_REGION_POOL_SIZE];
	__shared__ Region<NDIM>* gPool;
		
    Region<NDIM>* gRegionPool = 0;
    int sRegionPoolSize = INIT_REGION_POOL<IntegT, T, NDIM>(d_integrand, dRegions,
                                              dRegionsLength,
                                              subDividingDimension,
                                              numRegions,
                                              &constMem,
                                              FEVAL,
                                              NSETS,
											  sRegionPool,
											  gPool);

    ComputeErrResult<T, NDIM>(ERR, RESULT, sRegionPool);
    // TODO : May be redundance sync
    __syncthreads();

    int nregions = sRegionPoolSize; // is only 1 at this point

    while (nregions <= MAX_GLOBALPOOL_SIZE &&
           ERR > MaxErr(RESULT, epsrel, epsabs)) {
			   
      gRegionPool = gPool;
      sRegionPoolSize =
      EXTRACT_MAX<T, NDIM>(sRegionPool, gRegionPool, sRegionPoolSize, gpuId, gPool);
      Region<NDIM> *RegionLeft, *RegionRight;
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

        // TODO: What does div do!
        RegionRight->div = ++RegionLeft->div;
        for (int dim = 0; dim < NDIM; ++dim) {
          RegionRight->bounds[dim].lower = RegionLeft->bounds[dim].lower;
          RegionRight->bounds[dim].upper = RegionLeft->bounds[dim].upper;
        }
        // Subdivide the chosen axis
        bL->upper = bR->lower = 0.5 * (bL->lower + bL->upper);
      }

      sRegionPoolSize++;

      __syncthreads();
      SampleRegionBlock<IntegT, T, NDIM>(d_integrand, 0, &constMem, FEVAL, NSETS, sRegionPool);
      __syncthreads();
      SampleRegionBlock<IntegT, T, NDIM>(d_integrand, sRegionPoolSize - 1, &constMem, FEVAL, NSETS, sRegionPool);
      __syncthreads();

      // update ERR & RESULT
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

        ERR += rL->err + rR->err - result.err;
        RESULT += rL->avg + rR->avg - result.avg;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {

      int isActive = ERR > MaxErr(RESULT, epsrel, epsabs);

      if (ERR > (1e+10)) {
        RESULT = 0.0;
        ERR = 0.0;
        isActive = 1;
      }

      activeRegions[blockIdx.x] = isActive;
      dRegionsIntegral[blockIdx.x] = RESULT;
      dRegionsError[blockIdx.x] = ERR;
      dRegionsNumRegion[blockIdx.x] = nregions;

      free(gPool);
    }
  }
}

#endif

