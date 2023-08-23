#ifndef KOKKOS_CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH
#define KOKKOS_CUDACUHRE_QUAD_GPUQUAD_PHASES_CUH

#include "kokkos/pagani/quad/GPUquad/Sample.cuh"
#include "kokkos/pagani/quad/GPUquad/Func_Eval.cuh"
#include "kokkos/pagani/quad/quad.h"
#include "common/kokkos/Volume.cuh"

#define FINAL 0
// added for variadic cuprintf
#include <stdarg.h>
#include <stdio.h>
namespace quad {

  template <typename T>
  KOKKOS_INLINE_FUNCTION void
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
          T d = va_arg(args, T);
          printf("%f\n", d);
        }
        ++fmt;
      }
    }
    va_end(args);
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION T
  ScaleValue(T val, T min, T max)
  {
    // assert that max > min
    T range = fabs(max - min);
    return min + val * range;
  }

  template <typename T, int NDIM>
  KOKKOS_INLINE_FUNCTION void
  ActualCompute(Kokkos::View<T*, Kokkos::CudaSpace> generators,
                T* g,
                const Structures<double>& constMem,
                size_t feval_index,
                size_t total_feval,
                const member_type team_member)
  {

    for (int dim = 0; dim < NDIM; ++dim) {
      g[dim] = 0;
    }

    int threadIdx = team_member.team_rank();
    int blockIdx = team_member.league_rank();

    int posCnt = constMem.gpuGenPermVarStart(feval_index + 1) -
                 constMem.gpuGenPermVarStart(feval_index);
    int gIndex = constMem.gpuGenPermGIndex(feval_index);

    for (int posIter = 0; posIter < posCnt; ++posIter) {
      int pos = constMem.gpuGenPos((constMem.gpuGenPermVarStart(feval_index)) +
                                   posIter);
      int absPos = abs(pos);

      if (pos == absPos) {
        g[absPos - 1] = constMem.gpuG(gIndex * NDIM + posIter);
      } else {
        g[absPos - 1] = -constMem.gpuG(gIndex * NDIM + posIter);
      }
    }

    for (int dim = 0; dim < NDIM; dim++) {
      generators(total_feval * dim + feval_index) = g[dim];
    }
  }

  template <typename T, int NDIM>
  void
  ComputeGenerators(Kokkos::View<T*, Kokkos::CudaSpace> generators,
                    size_t FEVAL,
                    const Structures<double> constMem)
  {
    uint32_t nBlocks = 1;
    uint32_t nThreads = 64;

    Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> team_policy1(nBlocks,
                                                                  nThreads);
    auto team_policy = Kokkos::Experimental::require(
      team_policy1, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for(
      "Phase1", team_policy, KOKKOS_LAMBDA(const member_type team_member) {
        int threadIdx = team_member.team_rank();
        int blockIdx = team_member.league_rank();

        size_t perm = 0;
        T g[NDIM];

        for (size_t dim = 0; dim < NDIM; ++dim) {
          g[dim] = 0;
        }

        size_t feval_index = perm * nThreads + threadIdx;
        if (feval_index < FEVAL) {
          ActualCompute<T, NDIM>(
            generators, g, constMem, feval_index, FEVAL, team_member);
        }

        team_member.team_barrier();

        for (perm = 1; perm < FEVAL / nThreads; ++perm) {
          int feval_index = perm * nThreads + threadIdx;
          ActualCompute<T, NDIM>(
            generators, g, constMem, feval_index, FEVAL, team_member);
        }

        team_member.team_barrier();

        feval_index = perm * nThreads + threadIdx;
        if (feval_index < FEVAL) {
          int feval_index = perm * nThreads + threadIdx;
          ActualCompute<T, NDIM>(
            generators, g, constMem, feval_index, FEVAL, team_member);
        }

        team_member.team_barrier();
      });
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION void
  RefineError(T* dRegionsIntegral,
              T* dRegionsError,
              T* dParentsIntegral,
              T* dParentsError,
              T* newErrs,
              int* activeRegions,
              size_t currIterRegions,
              T epsrel,
              int heuristicID)
  {
    size_t numThreads = 64;
    size_t numBlocks =
      currIterRegions / numThreads + ((currIterRegions % numThreads) ? 1 : 0);

    Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> team_policy1(numBlocks,
                                                                  numThreads);
    auto team_policy = Kokkos::Experimental::require(
      team_policy1, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for(
      "RefineError", team_policy, KOKKOS_LAMBDA(const member_type team_member) {
        int threadIdx = team_member.team_rank();
        int blockIdx = team_member.league_rank();

        // can we do anythign with the rest of the threads? maybe launch more
        // blocks instead and a  single thread per block?
        size_t tid = blockIdx * numThreads + threadIdx;

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

          T diff = siblRes + selfRes - parRes;
          diff = fabs(.25 * diff);

          T err = selfErr + siblErr;

          if (err > 0.0) {
            T c = 1 + 2 * diff / err;
            selfErr *= c;
          }

          selfErr += diff;

          newErrs[tid] = selfErr;
          int PassRatioTest =
            heuristicID != 1 &&
            selfErr < MaxErr(selfRes, epsrel, /*epsabs*/ 1e-200);
          activeRegions[tid] = !(/*polished ||*/ PassRatioTest);
        }
      });
  }

  KOKKOS_INLINE_FUNCTION void
  RevertFinishedStatus(int* activeRegions, size_t numRegions)
  {

    size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numRegions) {
      activeRegions[tid] = 1;
    }
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION void
  Filter(T const* dRegionsError,
         int* unpolishedRegions,
         int const* activeRegions,
         size_t numRegions,
         T errThreshold)
  {
    size_t const tid = blockIdx.x * blockDim.x + threadIdx.x;

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
  KOKKOS_INLINE_FUNCTION void
  INIT_REGION_POOL(IntegT* d_integrand,
                   T* dRegions,
                   T* dRegionsLength,
                   size_t numRegions,
                   const Structures<T>& constMem,
                   T* lows,
                   T* highs,
                   T* generators,
                   Region<NDIM>* sRegionPool,
                   quad::Func_Evals<NDIM> fevals,
                   const member_type team_member)
  {
    typedef Kokkos::View<Region<NDIM>*,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ScratchViewRegion;

    SampleRegionBlock<IntegT, T, NDIM, blockDim, debug>(d_integrand,
                                                        constMem,
                                                        sRegionPool,
                                                        dRegions,
                                                        dRegionsLength,
                                                        numRegions,
                                                        lows,
                                                        highs,
                                                        generators,
                                                        fevals,
                                                        team_member);
    team_member.team_barrier();
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
    Structures<T> constMem, // switch to const ptr:  Structures<double> const *
                            // const constMem,
    T* lows,
    T* highs,
    T* generators,
    quad::Func_Evals<NDIM> fevals)
  {

    uint32_t nBlocks = numRegions;
    uint32_t nThreads = blockDim;
    typedef Kokkos::View<Region<NDIM>*,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ScratchViewRegion;

    Kokkos::TeamPolicy<> mainKernelPolicy(nBlocks,
                                                                      nThreads);
   
    int shMemBytes =
      ScratchViewRegion::shmem_size(1) +
      ScratchViewDouble::shmem_size(nThreads);  // for sdata

    Kokkos::parallel_for(
      "INTEGRATE_GPU_PHASE1",
      mainKernelPolicy.set_scratch_size(0, Kokkos::PerTeam(shMemBytes)),
      KOKKOS_LAMBDA(const member_type team_member) {

        ScratchViewRegion sRegionPool(team_member.team_scratch(0), 1);
        INIT_REGION_POOL<IntegT, T, NDIM, blockDim, debug>(d_integrand,
                                                           dRegions,
                                                           dRegionsLength,
                                                           numRegions,
                                                           constMem,
                                                           lows,
                                                           highs,
                                                           generators,
                                                           sRegionPool.data(),
                                                           fevals,
                                                           team_member);

        team_member.team_barrier();

        if (team_member.team_rank() == 0) {
          subDividingDimension[team_member.league_rank()] =
            sRegionPool(0).result.bisectdim;
          dRegionsIntegral[team_member.league_rank()] =
            sRegionPool(0).result.avg;
          dRegionsError[team_member.league_rank()] = sRegionPool(0).result.err;
        }
      });
  }

  __device__ size_t
  GetSiblingIndex(size_t numRegions)
  {
    return (2 * blockIdx.x / numRegions) < 1 ? blockIdx.x + numRegions :
                                               blockIdx.x - numRegions;
  }
}

#endif
