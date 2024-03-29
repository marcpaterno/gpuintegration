#ifndef CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH
#define CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH

#include <CL/sycl.hpp>
#include "oneAPI/pagani/quad/GPUquad/Sample.dp.hpp"
#include "common/oneAPI/Volume.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Func_Eval.hpp"

#define FINAL 0
// added for variadic cuprintf
#include <stdarg.h>
#include <stdio.h>
namespace quad {

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
                  Structures<T>* constMem,
                  int FEVAL,
                  int NSETS,
                  sycl::nd_item<3> item_ct1)
  {
    T ERR = 0, RESULT = 0;
    INIT_REGION_POOL(
      dRegions, dRegionsLength, numRegions, constMem, FEVAL, NSETS);

    if (item_ct1.get_local_id(2) == 0) {
      dRegionsIntegral[item_ct1.get_group(2)] =
        sRegionPool[item_ct1.get_local_id(2)].result.avg;
      dRegionsError[item_ct1.get_group(2)] =
        sRegionPool[item_ct1.get_local_id(2)].result.err;

      item_ct1.barrier();
    }
  }

  template <typename T, int NDIM>
  void
  ActualCompute(double* generators,
                T* g,
                Structures<double> constMem,
                size_t feval_index,
                size_t total_feval)
  {
    for (int i = 0; i < NDIM; ++i)
      g[i] = 0.;

    int posCnt = constMem._gpuGenPermVarStart[feval_index + 1] -
                 constMem._gpuGenPermVarStart[feval_index];

    int gIndex = constMem._gpuGenPermGIndex[feval_index];

    for (int posIter = 0; posIter < posCnt; ++posIter) {
      int pos =
        (constMem
           ._gpuGenPos[(constMem._gpuGenPermVarStart[feval_index]) + posIter]);
      int absPos = sycl::abs(pos);

      if (pos == absPos) {
        g[absPos - 1] = constMem._gpuG[gIndex * NDIM + posIter];
      } else {
        g[absPos - 1] = -constMem._gpuG[gIndex * NDIM + posIter];
      }
    }

    for (int dim = 0; dim < NDIM; dim++) {
      generators[total_feval * dim + feval_index] = g[dim];
    }
  }

  template <typename T, int NDIM>
  void
  ComputeGenerators(double* generators,
                    size_t FEVAL,
                    Structures<double> constMem,
                    sycl::nd_item<1> item_ct1)
  {
    size_t perm = 0;
    T g[NDIM] = {0.};
    int block_size = 64;

    size_t feval_index = perm * block_size + item_ct1.get_local_id(0);
    if (feval_index < FEVAL) {
      ActualCompute<T, NDIM>(generators, g, constMem, feval_index, FEVAL);
    }

    item_ct1.barrier();
    for (perm = 1; perm < FEVAL / block_size; ++perm) {
      int feval_index = perm * block_size + item_ct1.get_local_id(0);
      ActualCompute<T, NDIM>(generators, g, constMem, feval_index, FEVAL);
    }

    item_ct1.barrier();
    feval_index = perm * block_size + item_ct1.get_local_id(0);
    if (feval_index < FEVAL) {
      ActualCompute<T, NDIM>(generators, g, constMem, feval_index, FEVAL);
    }

    item_ct1.barrier();
  }

  template <typename T>
  void
  RefineError(T* dRegionsIntegral,
              T* dRegionsError,
              T* dParentsIntegral,
              T* dParentsError,
              T* newErrs,
              double* activeRegions,
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
      int PassRatioTest =
        heuristicID != 1 && selfErr < MaxErr(selfRes, epsrel, 1e-200);
      activeRegions[tid] = static_cast<double>(!(PassRatioTest));
    }
  }

  void
  RevertFinishedStatus(int* activeRegions,
                       size_t numRegions,
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

  template <typename IntegT, typename T, int NDIM, int blockDim, int debug = 0>
  void
  INIT_REGION_POOL(IntegT* d_integrand,
                   T* dRegions,
                   T* dRegionsLength,
                   size_t numRegions,
                   Structures<double>& constMem,
                   Region<NDIM> sRegionPool[],
                   GlobalBounds sBound[],
                   T* lows,
                   T* highs,
                   double* generators,
                   sycl::nd_item<1> item_ct1,
                   T* shared,
                   T* sdata,
                   T* Jacobian,
                   int* maxDim,
                   T* vol,
                   T* ranges,
                   quad::Func_Evals<NDIM>& fevals)
  {
    const size_t index = item_ct1.get_group(0);
    // may not be worth pre-computing

    if (item_ct1.get_local_id(0) == 0) {

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
    item_ct1.barrier();
    SampleRegionBlock<IntegT, T, NDIM, blockDim, debug>(d_integrand,
                                                        // 0,
                                                        constMem,
                                                        // FEVAL,
                                                        // NSETS,
                                                        sRegionPool,
                                                        sBound,
                                                        vol,
                                                        maxDim,
                                                        ranges,
                                                        Jacobian,
                                                        generators,
                                                        item_ct1,
                                                        shared,
                                                        sdata,
                                                        fevals);
    item_ct1.barrier();
  }

  template <typename IntegT, typename T, int NDIM, int blockDim, int debug = 0>
  void
  INTEGRATE_GPU_PHASE1(IntegT* d_integrand,
                       T* dRegions,
                       T* dRegionsLength,
                       size_t numRegions,
                       T* dRegionsIntegral,
                       T* dRegionsError,
                       // double* activeRegions,
                       int* subDividingDimension,
                       T epsrel,
                       T epsabs,
                       Structures<double> constMem,
                       T* lows,
                       T* highs,
                       double* generators,
                       sycl::nd_item<1> item_ct1,
                       T* shared,
                       T* sdata,
                       T* Jacobian,
                       int* maxDim,
                       T* vol,
                       T* ranges,
                       Region<NDIM>* sRegionPool,
                       GlobalBounds* sBound,
                       quad::Func_Evals<NDIM> fevals)
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
                                                       item_ct1,
                                                       shared,
                                                       sdata,
                                                       Jacobian,
                                                       maxDim,
                                                       vol,
                                                       ranges,
                                                       fevals);

    if (item_ct1.get_local_id(0) == 0) {
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
}

#endif
