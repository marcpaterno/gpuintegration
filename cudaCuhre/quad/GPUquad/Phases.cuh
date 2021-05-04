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
   
   __device__ bool 
   ApplyHeuristic(int heuristicID, 
                            double leaves_estimate, 
                            double finished_estimate, 
                            double queued_estimate, 
                            double lastErr, 
                            double finished_errorest, 
                            double queued_errorest, 
                            size_t currIterRegions, 
                            size_t total_nregions,
                            bool minIterReached,
                            double parErr,
                            double parRes,
                            int depth,
                            double selfRes,
                            double selfErr,
                            double epsrel,
                            double epsabs){
       
        double GlobalErrTarget = fabs(leaves_estimate) * epsrel;
        double remainGlobalErrRoom = GlobalErrTarget - finished_errorest - queued_errorest;
        bool selfErrTarget = fabs(selfRes) * epsrel;
        
        bool worstCaseScenarioGood;
        
        auto ErrBiggerThanEstimateCase = [selfRes, selfErr, parRes, parErr, remainGlobalErrRoom, currIterRegions](){
           return selfErr > fabs(selfRes) && selfErr/fabs(selfRes) >= .9*parErr/fabs(parRes) &&  selfErr < remainGlobalErrRoom/currIterRegions ;
        };
        
        
        switch(heuristicID){
           case 0:
                worstCaseScenarioGood = false;
                break;          
            case 1:
                worstCaseScenarioGood = false; 
                break;
            case 2: //useless right now, same as heuristic 1
                worstCaseScenarioGood = 
                ErrBiggerThanEstimateCase()
                || 
                 (selfRes < (leaves_estimate * epsrel * depth)/(total_nregions) && selfErr*currIterRegions < remainGlobalErrRoom);       
                break;          
            case 4:
                worstCaseScenarioGood = 
                ErrBiggerThanEstimateCase()
                || 
                 (fabs(selfRes) < (fabs(leaves_estimate) * epsrel * depth)/(total_nregions) && 
                 selfErr*currIterRegions < GlobalErrTarget);
                break;           
            case 7:
                worstCaseScenarioGood = 
                 (selfRes*currIterRegions + queued_estimate + finished_estimate < leaves_estimate && selfErr*currIterRegions < GlobalErrTarget);
                 break;       
            case 8:
                worstCaseScenarioGood = selfRes < leaves_estimate/total_nregions || 
                                        selfErr < epsrel*leaves_estimate/total_nregions;
                break;
            case 9:
                 worstCaseScenarioGood = selfRes < leaves_estimate/total_nregions && 
                                         selfErr < epsrel*leaves_estimate/total_nregions;
                break;
            case 10:
                worstCaseScenarioGood = fabs(selfRes) < 2*leaves_estimate/pow(2,depth) && 
                                         selfErr < 2*leaves_estimate*epsrel/pow(2,depth);
        }
              
        bool verdict = (worstCaseScenarioGood && minIterReached) || (selfRes == 0. && selfErr <= epsabs && minIterReached);
        return verdict;
   }
   
   template<int NDIM>
   __device__
   void
   ActualCompute(double* generators, double* g, const Structures<double>& constMem, size_t feval_index, size_t total_feval){
       for (int dim = 0; dim < NDIM; ++dim) {
        g[dim] = 0;
       }
        int posCnt = __ldg(&constMem._gpuGenPermVarStart[feval_index + 1]) -
                 __ldg(&constMem._gpuGenPermVarStart[feval_index]);
        int gIndex = __ldg(&constMem._gpuGenPermGIndex[feval_index]);   
   
        for (int posIter = 0; posIter < posCnt; ++posIter) {
          int pos =
            (constMem._gpuGenPos[(constMem._gpuGenPermVarStart[feval_index]) + posIter]);
          int absPos = abs(pos);
          
          if (pos == absPos) { 
            g[absPos - 1] = __ldg(&constMem._gpuG[gIndex * NDIM + posIter]);
          } else {
            g[absPos - 1] = -__ldg(&constMem._gpuG[gIndex * NDIM + posIter]);
          }
        }
        
        for(int dim=0; dim<NDIM; dim++){
            generators[total_feval*dim + feval_index] = g[dim];
        }
        
   }
 
 template<int NDIM>
  __global__
  void
  ComputeGenerators(double* generators, size_t FEVAL, const Structures<double> constMem){
    size_t perm = 0;
    double g[NDIM];
    for (size_t dim = 0; dim < NDIM; ++dim) {
      g[dim] = 0;
    }
    
    size_t feval_index = perm * BLOCK_SIZE + threadIdx.x;
    //printf("[%i] Processing feval_index:%i\n", threadIdx.x, feval_index);
    if (feval_index < FEVAL) {
     ActualCompute<NDIM>(generators, g, constMem, feval_index, FEVAL);
    }
    __syncthreads();
    for (perm = 1; perm < FEVAL / BLOCK_SIZE; ++perm) {
       int feval_index = perm * BLOCK_SIZE + threadIdx.x;
       ActualCompute<NDIM>(generators, g, constMem, feval_index, FEVAL);
    }
    __syncthreads();
    feval_index = perm * BLOCK_SIZE + threadIdx.x;
    if (feval_index < FEVAL) {
      int feval_index = perm * BLOCK_SIZE + threadIdx.x;
      ActualCompute<NDIM>(generators, g, constMem, feval_index, FEVAL);
    }
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
              size_t currIterRegions,
              //size_t total_nregions,
              //double iterEstimate,
              //double leaves_estimate,
              //double finished_estimate,
              //double finished_errorest,
              //double queued_estimate,
              //double queued_errorest,
              T epsrel,
              //T epsabs,
              //int depth,
              //bool estConverged,
              //double lastErr,
              int heuristicID)
  {
    // can we do anythign with the rest of the threads? maybe launch more blocks
    // instead and a  single thread per block?
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid<currIterRegions){ 
      T selfErr = dRegionsError[tid];
      T selfRes = dRegionsIntegral[tid];
      
      size_t inRightSide = (2*tid >= currIterRegions);
      size_t inLeftSide =  (0 >= inRightSide);
      size_t siblingIndex = tid  + (inLeftSide*currIterRegions/2) - (inRightSide*currIterRegions/2);
      size_t parIndex = tid - inRightSide*(currIterRegions*.5);
        
      T siblErr = dRegionsError[siblingIndex];
      T siblRes = dRegionsIntegral[siblingIndex];
        
      T parRes = dParentsIntegral[parIndex];
      //T parErr = dParentsError[parIndex];
              
      T diff = siblRes + selfRes - parRes;
      diff = fabs(.25 * diff);

      T err = selfErr + siblErr;
        
      if (err > 0.0) {
        T c = 1 + 2 * diff / err;
        selfErr *= c;
      }

      selfErr += diff;

      newErrs[tid] = selfErr;  
      /*int polished = ApplyHeuristic(heuristicID, 
                             leaves_estimate, 
                             finished_estimate, 
                             queued_estimate, 
                             lastErr, 
                             finished_errorest, 
                             queued_errorest, 
                             currIterRegions, 
                             total_nregions,
                             estConverged,
                             parErr,
                             parRes,
                             depth,
                             selfRes,
                             selfErr,
                             epsrel,
                             epsabs) == true;*/
      int PassRatioTest = heuristicID != 1 && selfErr < MaxErr(selfRes, epsrel, /*epsabs*/1e-200);
      activeRegions[tid] = !(/*polished ||*/ PassRatioTest);
      
      
    }
  }

  __global__ void
  RevertFinishedStatus(int* activeRegions, size_t numRegions){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;        
     if(tid<numRegions){
        activeRegions[tid] = 1;
     }         
  }

    template <typename T>
  __global__ void
  Filter( T* dRegionsError,
         int* unpolishedRegions,
         int* activeRegions,
         size_t numRegions,
         double errThreshold)
  {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid<numRegions){ //consider not having the ones passing the previous test (id<numRegions && activeRegions[tid] != 1)
      
      T selfErr = dRegionsError[tid];
      unpolishedRegions[tid] = (selfErr > errThreshold ) * activeRegions[tid]; //onle "real active" regions can be polished (rename activeRegions in this context to polishedRegions)
      
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
                   int depth,
                   double* generators)
  {
    size_t index = blockIdx.x;
    __shared__ double vol;
    __shared__ int maxDim;
    __shared__ double ranges[NDIM];
    __shared__ double Jacobian;
    
    if (threadIdx.x == 0) {
      Jacobian = 1;
      double maxRange = 0;
      for (int dim = 0; dim < NDIM; ++dim) {
        T lower = dRegions[dim * numRegions + index];
        sRegionPool[0].bounds[dim].lower = lower;
        sRegionPool[0].bounds[dim].upper = 
          lower + dRegionsLength[dim * numRegions + index];
        
        sBound[dim].unScaledLower = lows[dim];
        sBound[dim].unScaledUpper = highs[dim];
        ranges[dim] = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;
        sRegionPool[0].div = depth; 
        
        double range = sRegionPool[0].bounds[dim].upper - lower;
        Jacobian = Jacobian * ranges[dim];
        if(range > maxRange){
            maxDim = dim;
            maxRange = range;
        }
      }
      vol = ldexp(1., -depth);
    }

    __syncthreads();
    SampleRegionBlock<IntegT, T, NDIM>(
      d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool, &vol, &maxDim, ranges, &Jacobian, generators);
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
                       int depth,
                       double* generators)
  {
    __shared__ Region<NDIM> sRegionPool[1];

    INIT_REGION_POOL<IntegT>(d_integrand,
                             dRegions,
                             dRegionsLength,
                             numRegions,
                             constMem,
                             FEVAL,
                             NSETS,
                             sRegionPool,
                             lows,
                             highs,
                             iteration,
                             depth, generators);
    
    if (threadIdx.x == 0) {
      const double ERR = sRegionPool[0].result.err;
      const double RESULT = sRegionPool[0].result.avg;
        
      activeRegions[blockIdx.x] = 1; 
      subDividingDimension[blockIdx.x] = sRegionPool[0].result.bisectdim;
      dRegionsIntegral[blockIdx.x] = RESULT;
      dRegionsError[blockIdx.x] = ERR;
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
  InitSMemRegions(Region<NDIM> sRegionPool[], int depth)
  {
    int idx = 0;
    for (; idx < SM_REGION_POOL_SIZE / BLOCK_SIZE; ++idx) {

      int index = idx * BLOCK_SIZE + threadIdx.x;
      sRegionPool[index].div = depth;
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

      sRegionPool[index].div = depth;
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
        T lower = dRegions[dim * numRegions + intervalIndex];
        sRegionPool[threadIdx.x].bounds[dim].lower = lower;
        sRegionPool[threadIdx.x].bounds[dim].upper = lower + dRegionsLength[dim * numRegions + intervalIndex];
        
        sBound[dim].unScaledLower = lows[dim];
        sBound[dim].unScaledUpper = highs[dim];
      }
    }
    __syncthreads();
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
                             double* generators,
                             int depth = 0,
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
    
    int maxdiv = 0;
    // SetBlockVars<T, NDIM>(slows, shighs, max_global_pool_size, gPool, lows,
    // highs, max_regions, ggRegionPool);

    if (threadIdx.x == 0) {
      memcpy(slows, lows, sizeof(T) * NDIM);
      memcpy(shighs, highs, sizeof(T) * NDIM);
      max_global_pool_size = max_regions;
      gPool = &ggRegionPool[blockIdx.x * max_regions];
    }

    __syncthreads();              // added for testing
    InitSMemRegions(sRegionPool, depth); // sets every region in shared memory to zero
    int sRegionPoolSize = 1;
    __syncthreads();
    // temporary solution

    SET_FIRST_SHARED_MEM_REGION(sRegionPool,
                                  batch.dRegions,
                                  batch.dRegionsLength,
                                  slows,
                                  shighs,
                                  batch.numRegions,
                                  blockIdx.x);


    __syncthreads();
    // ERR and sRegionPool[0].result.err are not the same in the beginning
    SampleRegionBlock_ph2<IntegT, T, NDIM>(d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool, generators);//, 0, 0, 0, 0, nullptr);
    ALIGN_GLOBAL_TO_SHARED<IntegT, T, NDIM>(sRegionPool, gPool);
    ComputeErrResult<T, NDIM>(ERR, RESULT, sRegionPool);
    
    // temporary solution
    if (threadIdx.x == 0) {
      ERR = batch.dRegionsError[blockIdx.x];
      sRegionPool[0].result.err = ERR;    
    }

    __syncthreads();
    
    int nregions = sRegionPoolSize; // is only 1 at this point
    while (nregions < max_global_pool_size && (nregions == 1 || ERR > MaxErr(RESULT, epsrel, /*1e-200*/epsabs))) {

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
      SampleRegionBlock_ph2<IntegT, T, NDIM>(
        d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool, generators);//, 0, 0, 0, 0, nullptr);
      __syncthreads();
      SampleRegionBlock_ph2<IntegT, T, NDIM>(
        d_integrand, sRegionPoolSize - 1, constMem, FEVAL, NSETS, sRegionPool, generators);//, 0, 0, 0, 0, nullptr);
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

        ERR += rL->err + rR->err - result.err;
        RESULT += rL->avg + rR->avg - result.avg;
      }
      __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0) {

      int isActive = ERR > MaxErr(RESULT, epsrel, epsabs/*1e-200*/);

      if (ERR > (1e+10)) {
        RESULT = 0.0;
        ERR = 0.0;
        isActive = 1;
      }
      
      //if(blockIdx.x == 0 || blockIdx.x == 32000)
      //  printf("[%i] %.15e %.15e\n", blockIdx.x, RESULT , batch.dRegionsIntegral[blockIdx.x]);
      batch.activeRegions[blockIdx.x] = isActive;
      batch.dRegionsIntegral[blockIdx.x] = RESULT;
      batch.dRegionsError[blockIdx.x] = ERR;
      dRegionsNumRegion[blockIdx.x] = nregions;
    }
  }


   
  //------------------------------------------------------------------------------------------
  //Dummy methods
  //Contain removed kernel code for comparison against Kokkos 
  
  
    template <typename IntegT, typename T, int NDIM>
  __device__ void
  dummyINIT_REGION_POOL(IntegT* d_integrand,
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
                   int depth,
                   double* generators)
  {
    size_t index = blockIdx.x;
    __shared__ double vol;
    __shared__ int maxDim;
    __shared__ double ranges[NDIM];
    __shared__ double Jacobian;
    
    if (threadIdx.x == 0) {
      Jacobian = 1;
      double maxRange = 0;
      for (int dim = 0; dim < NDIM; ++dim) {
        T lower = dRegions[dim * numRegions + index];
        sRegionPool[0].bounds[dim].lower = lower;
        sRegionPool[0].bounds[dim].upper = 
          lower + dRegionsLength[dim * numRegions + index];
        
        sBound[dim].unScaledLower = lows[dim];
        sBound[dim].unScaledUpper = highs[dim];
        ranges[dim] = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;
        sRegionPool[0].div = depth; 
        
        double range = sRegionPool[0].bounds[dim].upper - lower;
        Jacobian = Jacobian * ranges[dim];
        if(range > maxRange){
            maxDim = dim;
            maxRange = range;
        }
      }
      vol = ldexp(1., -depth);
    }

    __syncthreads();
    /*SampleRegionBlock<IntegT, T, NDIM>(
      d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool, &vol, &maxDim, ranges, &Jacobian, generators);*/
    __syncthreads();
  }
  
  template <typename IntegT, typename T, int NDIM>
  __global__ void
  dummyINTEGRATE_GPU_PHASE1(IntegT* d_integrand,
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
                       int depth,
                       double* generators)
  {
    __shared__ Region<NDIM> sRegionPool[1];
    
    dummyINIT_REGION_POOL<IntegT>(d_integrand,
                             dRegions,
                             dRegionsLength,
                             numRegions,
                             constMem,
                             FEVAL,
                             NSETS,
                             sRegionPool,
                             lows,
                             highs,
                             iteration,
                             depth, generators);
    __syncthreads();
    if (threadIdx.x == 0) {
      const double ERR = sRegionPool[0].result.err;
      const double RESULT = sRegionPool[0].result.avg;
        
      activeRegions[blockIdx.x] = 1; 
      subDividingDimension[blockIdx.x] = 0;//sRegionPool[0].result.bisectdim;
      dRegionsIntegral[blockIdx.x] = 1.;//RESULT;
      dRegionsError[blockIdx.x] = 1000.*epsrel;//ERR;
    }
  }
}

#endif
