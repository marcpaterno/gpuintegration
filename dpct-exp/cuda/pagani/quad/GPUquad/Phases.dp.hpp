#ifndef CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH
#define CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "dpct-exp/cuda/pagani/quad/GPUquad/Sample.dp.hpp"
#include "dpct-exp/common/cuda/Volume.dp.hpp"

#define FINAL 0
// added for variadic cuprintf
#include <stdarg.h>
#include <stdio.h>
namespace quad {

  template <typename T>
  /*
  DPCT1080:77: Variadic functions cannot be called in a SYCL kernel or by
  functions called by the kernel. You may need to adjust the code.
  */
  void
  cuprintf(const char* fmt,
           sycl::nd_item<3> item_ct1,
           const sycl::stream& stream_ct1,
           ...)
  {
    va_list args;
    va_start(args, fmt);

    if (item_ct1.get_local_id(2) == 0) {
      while (*fmt != '\0') {
        if (*fmt == 'd') {
          int i = va_arg(args, int);
          /*
          DPCT1015:78: Output needs adjustment.
          */
          stream_ct1 << "%d\n";
        } else if (*fmt == 'c') {
          int c = va_arg(args, int);
          /*
          DPCT1015:79: Output needs adjustment.
          */
          stream_ct1 << "%c\n";
        } else if (*fmt == 'f') {
          T d = va_arg(args, T);
          printf("%f\n", d);
        }
        ++fmt;
      }
    }
    va_end(args);
  }

  template <typename T>
  T
  ScaleValue(T val, T min, T max)
  {
    // assert that max > min
    T range = fabs(max - min);
    return min + val * range;
  }

  template <typename T, int NDIM>
  void
  QuickMassSample(T* dRegions,
                  T* dRegionsLength,
                  size_t numRegions,
                  Region<NDIM> sRegionPool[],
                  T* dRegionsIntegral,
                  T* dRegionsError,
                  Structures<T> constMem,
                  int FEVAL,
                  int NSETS,
                  sycl::nd_item<3> item_ct1)
  {
    T ERR = 0, RESULT = 0;
    INIT_REGION_POOL(
      dRegions, dRegionsLength, numRegions, &constMem, FEVAL, NSETS);

    if (item_ct1.get_local_id(2) == 0) {
      dRegionsIntegral[item_ct1.get_group(2)] =
        sRegionPool[item_ct1.get_local_id(2)].result.avg;
      dRegionsError[item_ct1.get_group(2)] =
        sRegionPool[item_ct1.get_local_id(2)].result.err;
      /*
      DPCT1065:80: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
  }

  template <typename T>
  bool
  ApplyHeuristic(int heuristicID,
                 T leaves_estimate,
                 T finished_estimate,
                 T queued_estimate,
                 T lastErr,
                 T finished_errorest,
                 T queued_errorest,
                 size_t currIterRegions,
                 size_t total_nregions,
                 bool minIterReached,
                 T parErr,
                 T parRes,
                 int depth,
                 T selfRes,
                 T selfErr,
                 T epsrel,
                 T epsabs)
  {

    T GlobalErrTarget = fabs(leaves_estimate) * epsrel;
    T remainGlobalErrRoom =
      GlobalErrTarget - finished_errorest - queued_errorest;
    bool selfErrTarget = fabs(selfRes) * epsrel;

    bool worstCaseScenarioGood;

    auto ErrBiggerThanEstimateCase = [selfRes,
                                      selfErr,
                                      parRes,
                                      parErr,
                                      remainGlobalErrRoom,
                                      currIterRegions]() {
      return selfErr > fabs(selfRes) &&
             selfErr / fabs(selfRes) >= .9 * parErr / fabs(parRes) &&
             selfErr < remainGlobalErrRoom / currIterRegions;
    };

    switch (heuristicID) {
      case 0:
        worstCaseScenarioGood = false;
        break;
      case 1:
        worstCaseScenarioGood = false;
        break;
      case 2: // useless right now, same as heuristic 1
        worstCaseScenarioGood =
          ErrBiggerThanEstimateCase() ||
          (selfRes < (leaves_estimate * epsrel * depth) / (total_nregions) &&
           selfErr * currIterRegions < remainGlobalErrRoom);
        break;
      case 4:
        worstCaseScenarioGood =
          ErrBiggerThanEstimateCase() ||
          (fabs(selfRes) <
             (fabs(leaves_estimate) * epsrel * depth) / (total_nregions) &&
           selfErr * currIterRegions < GlobalErrTarget);
        break;
      case 7:
        worstCaseScenarioGood =
          (selfRes * currIterRegions + queued_estimate + finished_estimate <
             leaves_estimate &&
           selfErr * currIterRegions < GlobalErrTarget);
        break;
      case 8:
        worstCaseScenarioGood =
          selfRes < leaves_estimate / total_nregions ||
          selfErr < epsrel * leaves_estimate / total_nregions;
        break;
      case 9:
        worstCaseScenarioGood =
          selfRes < leaves_estimate / total_nregions &&
          selfErr < epsrel * leaves_estimate / total_nregions;
        break;
      case 10:
        worstCaseScenarioGood =
          fabs(selfRes) < 2 * leaves_estimate / sycl::pown((float)(2), depth) &&
          selfErr <
            2 * leaves_estimate * epsrel / sycl::pown((float)(2), depth);
    }

    bool verdict = (worstCaseScenarioGood && minIterReached) ||
                   (selfRes == 0. && selfErr <= epsabs && minIterReached);
    return verdict;
  }

  template <typename T, int NDIM>
  void
  ActualCompute(T* generators,
                T* g,
                const Structures<T>& constMem,
                size_t feval_index,
                size_t total_feval)
  {
    for (int dim = 0; dim < NDIM; ++dim) {
      g[dim] = 0.;
    }

    /*
    DPCT1026:81: The call to __ldg was removed because there is no
    correspoinding API in DPC++.
    */
    int posCnt = constMem.gpuGenPermVarStart[feval_index + 1] -
                 /*
                 DPCT1026:82: The call to __ldg was removed because there is no
                 correspoinding API in DPC++.
                 */
                 constMem.gpuGenPermVarStart[feval_index];
    /*
    DPCT1026:83: The call to __ldg was removed because there is no
    correspoinding API in DPC++.
    */
    int gIndex = constMem.gpuGenPermGIndex[feval_index];

    for (int posIter = 0; posIter < posCnt; ++posIter) {
      int pos =
        (constMem
           .gpuGenPos[(constMem.gpuGenPermVarStart[feval_index]) + posIter]);
      int absPos = sycl::abs(pos);

      if (pos == absPos) {
        /*
        DPCT1026:84: The call to __ldg was removed because there is no
        correspoinding API in DPC++.
        */
        g[absPos - 1] = constMem.gpuG[gIndex * NDIM + posIter];
      } else {
        /*
        DPCT1026:85: The call to __ldg was removed because there is no
        correspoinding API in DPC++.
        */
        g[absPos - 1] = -constMem.gpuG[gIndex * NDIM + posIter];
      }
    }

    for (int dim = 0; dim < NDIM; dim++) {
      generators[total_feval * dim + feval_index] = g[dim];
    }
  }

  template <typename T, int NDIM>
  void
  ComputeGenerators(T* generators, size_t FEVAL, const Structures<T> constMem,
                    sycl::nd_item<3> item_ct1)
  {
    size_t perm = 0;
    T g[NDIM] = {0.};
    /*for (size_t dim = 0; dim < NDIM; ++dim) {
      g[dim] = 0.;
    }*/

    size_t feval_index = perm * BLOCK_SIZE + item_ct1.get_local_id(2);
    // printf("[%i] Processing feval_index:%i\n", threadIdx.x, feval_index);
    if (feval_index < FEVAL) {
      ActualCompute<T, NDIM>(generators, g, constMem, feval_index, FEVAL);
    }
    /*
    DPCT1065:86: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (perm = 1; perm < FEVAL / BLOCK_SIZE; ++perm) {
      int feval_index = perm * BLOCK_SIZE + item_ct1.get_local_id(2);
      ActualCompute<T, NDIM>(generators, g, constMem, feval_index, FEVAL);
    }
    /*
    DPCT1065:87: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    feval_index = perm * BLOCK_SIZE + item_ct1.get_local_id(2);
    if (feval_index < FEVAL) {
      int feval_index = perm * BLOCK_SIZE + item_ct1.get_local_id(2);
      ActualCompute<T, NDIM>(generators, g, constMem, feval_index, FEVAL);
    }
    /*
    DPCT1065:88: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  template <typename T>
  void
  RefineError(T* dRegionsIntegral,
              T* dRegionsError,
              T* dParentsIntegral,
              T* dParentsError,
              T* newErrs,
              int* activeRegions,
              size_t currIterRegions,
              T epsrel,
              int heuristicID,
              sycl::nd_item<3> item_ct1)
  {
    // can we do anythign with the rest of the threads? maybe launch more blocks
    // instead and a  single thread per block?
    size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);

    if (tid < currIterRegions) {
      T selfErr = dRegionsError[tid];
      T selfRes = dRegionsIntegral[tid];

      size_t inRightSide = (2 * tid >= currIterRegions);
      size_t inLeftSide = (0 >= inRightSide);
      size_t siblingIndex = tid + (inLeftSide * currIterRegions / 2) -
                            (inRightSide * currIterRegions / 2);
      size_t parIndex = tid - inRightSide * (currIterRegions * .5);

      T siblErr = dRegionsError[siblingIndex];
      T siblRes = dRegionsIntegral[siblingIndex];

      T parRes = dParentsIntegral[parIndex];
      // T parErr = dParentsError[parIndex];

      T diff = siblRes + selfRes - parRes;
      diff = sycl::fabs(.25 * diff);

      T err = selfErr + siblErr;

      if (err > 0.0) {
        T c = 1 + 2 * diff / err;
        selfErr *= c;
      }

      selfErr += diff;

      newErrs[tid] = selfErr;
      int PassRatioTest = heuristicID != 1 &&
                          selfErr < MaxErr(selfRes, epsrel, /*epsabs*/ 1e-200);
      activeRegions[tid] = !(/*polished ||*/ PassRatioTest);
    }
  }

  void
  RevertFinishedStatus(int* activeRegions, size_t numRegions,
                       sycl::nd_item<3> item_ct1)
  {
    size_t const tid =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);

    if (tid < numRegions) {
      activeRegions[tid] = 1;
    }
  }

  template <typename T>
  void
  Filter(T const* dRegionsError,
         int* unpolishedRegions,
         int const* activeRegions,
         size_t numRegions,
         T errThreshold,
         sycl::nd_item<3> item_ct1)
  {
    size_t const tid =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);

    // consider not having the ones passing the previous test
    //    (id<numRegions && activeRegions[tid] != 1)
    if (tid < numRegions) {
      T const selfErr = dRegionsError[tid];
      int const factor = (selfErr > errThreshold);
      // only "real active" regions can be polished (rename activeRegions in
      // this context to polishedRegions)
      unpolishedRegions[tid] = factor * activeRegions[tid];
    }
  }

  template <typename IntegT, typename T, int NDIM, int blockDim, int debug>
  void
  INIT_REGION_POOL(IntegT* d_integrand,
                   T* dRegions,
                   T* dRegionsLength,
                   size_t numRegions,
                   Structures<T>& constMem,
                   Region<NDIM> sRegionPool[],
                   GlobalBounds sBound[],
                   T* lows,
                   T* highs,
                   T* generators,
                   quad::Func_Evals<NDIM>& fevals,
                   sycl::nd_item<1> item_ct1,
                   T/*Fix the type mannually*/ *shared,
                   T *sdata,
                   T *Jacobian,
                   int *maxDim,
                   T *vol,
                   T *ranges)
  {
    const size_t index = item_ct1.get_group(0);
    // may not be worth pre-computing

    if (item_ct1.get_local_id(0) == 0) {

      *Jacobian = 1.;
      *vol = 1.;
      T maxRange = 0;

#pragma unroll NDIM
      for (int dim = 0; dim < NDIM; ++dim) {
        T lower = dRegions[dim * numRegions + index];
        sRegionPool[0].bounds[dim].lower = lower;
        sRegionPool[0].bounds[dim].upper =
          lower + dRegionsLength[dim * numRegions + index];
        *vol *=
          sRegionPool[0].bounds[dim].upper - sRegionPool[0].bounds[dim].lower;

        sBound[dim].unScaledLower = lows[dim];
        sBound[dim].unScaledUpper = highs[dim];
        ranges[dim] = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;

        T range = sRegionPool[0].bounds[dim].upper - lower;
        *Jacobian = *Jacobian * ranges[dim];
        if (range > maxRange) {
          *maxDim = dim;
          maxRange = range;
        }
      }
    }

    /*
    DPCT1065:89: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    SampleRegionBlock<IntegT, T, NDIM, blockDim, debug>(d_integrand,
                                                        constMem,
                                                        sRegionPool,
                                                        sBound,
                                                        vol,
                                                        maxDim,
                                                        ranges,
                                                        Jacobian,
                                                        generators,
                                                        fevals,
                                                        item_ct1,
                                                        shared,
                                                        sdata);
    /*
    DPCT1065:90: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  template <typename IntegT, typename T, int NDIM, int blockDim, int debug = 0>
  void
  INTEGRATE_GPU_PHASE1(
    IntegT* d_integrand,
    T* dRegions,
    T* dRegionsLength,
    size_t numRegions,
    T* dRegionsIntegral,
    T* dRegionsError,
    int* subDividingDimension,
    T epsrel,
    T epsabs,
    Structures<T> constMem, // switch to const ptr:  Structures<double> const *
                            // const constMem,
    T* lows,
    T* highs,
    T* generators,
    quad::Func_Evals<NDIM> fevals,
    sycl::nd_item<1> item_ct1,
    T *shared,
    T *sdata,
    T *Jacobian,
    int *maxDim,
    T *vol,
    T *ranges,
    Region<NDIM> *sRegionPool,
    GlobalBounds *sBound)
  {

    INIT_REGION_POOL<IntegT, T, NDIM, blockDim, debug>(d_integrand,
                                                       dRegions,
                                                       dRegionsLength,
                                                       numRegions,
                                                       constMem,
                                                       sRegionPool,
                                                       sBound,
                                                       lows,
                                                       highs,
                                                       generators,
                                                       fevals,
                                                       item_ct1,
                                                       shared,
                                                       sdata,
                                                       Jacobian,
                                                       maxDim,
                                                       vol,
                                                       ranges);
    //if (item_ct1.get_local_id(0) == 0) 
	{
      subDividingDimension[item_ct1.get_group(0)] =
        sRegionPool[0].result.bisectdim;
      dRegionsIntegral[item_ct1.get_group(0)] = sRegionPool[0].result.avg;
      dRegionsError[item_ct1.get_group(0)] = sRegionPool[0].result.err;
	  
    }
  }

  size_t
  GetSiblingIndex(size_t numRegions, sycl::nd_item<3> item_ct1)
  {
    return (2 * item_ct1.get_group(2) / numRegions) < 1 ?
             item_ct1.get_group(2) + numRegions :
             item_ct1.get_group(2) - numRegions;
  }

  template <typename IntegT, typename T, int NDIM, int blockDim>
  void
  VEGAS_ASSISTED_INIT_REGION_POOL(IntegT* d_integrand,
                                  T* dRegions,
                                  T* dRegionsLength,
                                  size_t numRegions,
                                  Structures<T>& constMem,
                                  // int FEVAL,
                                  // int NSETS,
                                  Region<NDIM> sRegionPool[],
                                  GlobalBounds sBound[],
                                  T* lows,
                                  T* highs,
                                  T* generators,
                                  quad::Func_Evals<NDIM>& fevals,
                                  unsigned int seed_init,
                                  sycl::nd_item<3> item_ct1,
                                  T/*Fix the type mannually*/ *shared,
                                  T *xi,
                                  T *d,
                                  T *Jacobian,
                                  int *maxDim,
                                  T *vol,
                                  T *ranges,
                                  T *sdata)
  {
    size_t index = item_ct1.get_group(2);
    // may not be worth pre-computing

    if (item_ct1.get_local_id(2) == 0) {

      *Jacobian = 1.;
      *vol = 1.;
      T maxRange = 0;
      for (int dim = 0; dim < NDIM; ++dim) {
        T lower = dRegions[dim * numRegions + index];
        sRegionPool[0].bounds[dim].lower = lower;
        sRegionPool[0].bounds[dim].upper =
          lower + dRegionsLength[dim * numRegions + index];
        *vol *=
          sRegionPool[0].bounds[dim].upper - sRegionPool[0].bounds[dim].lower;

        sBound[dim].unScaledLower = lows[dim];
        sBound[dim].unScaledUpper = highs[dim];
        ranges[dim] = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;

        T range = sRegionPool[0].bounds[dim].upper - lower;
        *Jacobian = *Jacobian * ranges[dim];
        if (range > maxRange) {
          *maxDim = dim;
          maxRange = range;
        }
      }
    }

    /*
    DPCT1065:91: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    Vegas_assisted_SampleRegionBlock<IntegT, T, NDIM, blockDim>(d_integrand,
                                                                constMem,
                                                                sRegionPool,
                                                                sBound,
                                                                vol,
                                                                maxDim,
                                                                ranges,
                                                                Jacobian,
                                                                generators,
                                                                fevals,
                                                                seed_init,
                                                                item_ct1,
                                                                shared,
                                                                xi,
                                                                d,
                                                                sdata);
    /*
    DPCT1065:92: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  template <typename IntegT, typename T, int NDIM, int blockDim>
  void
  VEGAS_ASSISTED_INTEGRATE_GPU_PHASE1(IntegT* d_integrand,
                                      T* dRegions,
                                      T* dRegionsLength,
                                      size_t numRegions,
                                      T* dRegionsIntegral,
                                      T* dRegionsError,
                                      int* subDividingDimension,
                                      T epsrel,
                                      T epsabs,
                                      Structures<T> constMem,
                                      T* lows,
                                      T* highs,
                                      T* generators,
                                      quad::Func_Evals<NDIM> fevals,
                                      unsigned int seed_init,
                                      sycl::nd_item<3> item_ct1,
                                      T/*Fix the type mannually*/ *shared,
                                      T *xi,
                                      T *d,
                                      T *Jacobian,
                                      int *maxDim,
                                      T *vol,
                                      T *ranges,
                                      T *sdata,
                                      Region<NDIM> *sRegionPool,
                                      GlobalBounds *sBound)
  {

    VEGAS_ASSISTED_INIT_REGION_POOL<IntegT, T, NDIM, blockDim>(d_integrand,
                                                               dRegions,
                                                               dRegionsLength,
                                                               numRegions,
                                                               constMem,
                                                               sRegionPool,
                                                               sBound,
                                                               lows,
                                                               highs,
                                                               generators,
                                                               fevals,
                                                               seed_init,
                                                               item_ct1,
                                                               shared,
                                                               xi,
                                                               d,
                                                               Jacobian,
                                                               maxDim,
                                                               vol,
                                                               ranges,
                                                               sdata);

    if (item_ct1.get_local_id(2) == 0) {
      subDividingDimension[item_ct1.get_group(2)] =
        sRegionPool[0].result.bisectdim;
      dRegionsIntegral[item_ct1.get_group(2)] = sRegionPool[0].result.avg;
      dRegionsError[item_ct1.get_group(2)] = sRegionPool[0].result.err;
    }
  }

}

#endif
