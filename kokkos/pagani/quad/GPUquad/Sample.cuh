#ifndef KOKKOS_CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH
#define KOKKOS_CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH

#include <assert.h>
#include "kokkos/pagani/quad/quad.h"
#include "common/kokkos/Volume.cuh"
#include "common/kokkos/cudaApply.cuh"
#include "common/kokkos/cudaMemoryUtil.h"
#include "kokkos/pagani/quad/GPUquad/Func_Eval.cuh"
#include <cmath>

namespace quad {
  template <typename T>
  KOKKOS_INLINE_FUNCTION T
  Sq(T x)
  {
    return x * x;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION T
  warpReduceSum(T val)
  {
    val += __shfl_down_sync(0xffffffff, val, 16, 32);
    val += __shfl_down_sync(0xffffffff, val, 8, 32);
    val += __shfl_down_sync(0xffffffff, val, 4, 32);
    val += __shfl_down_sync(0xffffffff, val, 2, 32);
    val += __shfl_down_sync(0xffffffff, val, 1, 32);
    return val;
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION T
  blockReduceSum(T val, const member_type team_member)
  {
    double sum = 0.;
    team_member.team_barrier();
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, team_member.team_size()),
      [=](int& i, T& lsum) { lsum += val; },
      sum);
    return sum;
  }

  template <typename IntegT, typename T, int NDIM, int debug = 0>
  KOKKOS_INLINE_FUNCTION void
  computePermutation(IntegT* d_integrand,
                     int pIndex,
                     T* rlows,
                     T* rhighs,
                     T* global_lows,
                     T* sum,
                     const Structures<T>& constMem,
                     T* ranges,
                     T jacobian,
                     T* generators,
                     T* sdata,
                     quad::Func_Evals<NDIM> fevals,
                     const member_type team_member)
  {
    const int threadIdx = team_member.team_rank();
    gpu::cudaArray<T, NDIM> x;
    int blockIdx = team_member.league_rank();

    for (int dim = 0; dim < NDIM; ++dim) {
      const T generator =
        (generators[pagani::CuhreFuncEvalsPerRegion<NDIM>() * dim + pIndex]);
      x[dim] = global_lows[dim] + ((.5 + generator) * rlows[dim]+ (.5 - generator) * rhighs[dim]) * ranges[dim];
                                      
    }

    const T fun = gpu::apply(*d_integrand, x) * (jacobian);
    sdata[threadIdx] = fun;
    const int gIndex = (constMem.gpuGenPermGIndex[pIndex]);

    if constexpr (debug >= 2) {
      const int blockIdx = team_member.league_rank();
      // assert(fevals != nullptr);
      fevals[blockIdx * pagani::CuhreFuncEvalsPerRegion<NDIM>() + pIndex].store(
        x, global_lows, ranges, rlows, rhighs);
      fevals[blockIdx * pagani::CuhreFuncEvalsPerRegion<NDIM>() + pIndex].store(
        gpu::apply(*d_integrand, x), pIndex);
    }

    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * (constMem.cRuleWt[gIndex * NRULES + rul]);
    }
  }

  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
  template <typename IntegT, typename T, int NDIM, int blockdim, int debug = 0>
  KOKKOS_INLINE_FUNCTION void
  SampleRegionBlock(IntegT* d_integrand,
                    const Structures<T>& constMem,
                    Region<NDIM>* sRegionPool,
                    T* dRegions,
                    T* dRegionsLength,
                    int numRegions,
                    T* global_lows,
                    T* global_highs,
                    T* generators,
                    quad::Func_Evals<NDIM> fevals,
                    const member_type team_member)
  {


    int blockIdx = team_member.league_rank();
    double jacobian = 1.;
    double vol = 1.;
    double maxRange = 0;
    double rlows[NDIM];
    double rhighs[NDIM];
    double ranges[NDIM];
    int  maxDim;

    for (int dim = 0; dim < NDIM; ++dim) {
        const double lower = dRegions[dim * numRegions + blockIdx];
        rlows[dim] = lower;
        rhighs[dim] = lower + dRegionsLength[dim * numRegions + blockIdx];
        vol *= rhighs[dim] - rlows[dim];
        
        ranges[dim] = global_highs[dim] - global_lows[dim];
        jacobian *= ranges[dim];
        double range = rhighs[dim] - lower;

        if (range  > maxRange) {
          maxDim = dim;
          maxRange = range;
      }
    }
    
    const int threadIdx = team_member.team_rank();
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[0];
    ScratchViewDouble sdata(team_member.team_scratch(0), blockdim);
    int perm = 0;
    constexpr int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + threadIdx;
    constexpr int FEVAL = pagani::CuhreFuncEvalsPerRegion<NDIM>();
    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 rlows,
                                                 rhighs,
                                                 global_lows,
                                                 sum,
                                                 constMem,
                                                 ranges,
                                                 jacobian,
                                                 generators,
                                                 sdata.data(),
                                                 fevals,
                                                 team_member);
    }

    team_member.team_barrier();

    if (threadIdx == 0) {
      const T ratio =
        Sq(__ldg(&constMem.gpuG[2 * NDIM]) / __ldg(&constMem.gpuG[1 * NDIM]));
      T* f = &sdata[0];
      Result* r = &region->result;
      T* f1 = f;
      T base = *f1 * 2 * (1 - ratio);
      T maxdiff = 0;
      int bisectdim = maxDim;
      // #pragma unroll 1
      for (int dim = 0; dim < NDIM; ++dim) {
        T* fp = f1 + 1;
        T* fm = fp + 1;
        T fourthdiff =
          fabs(base + ratio * (fp[0] + fm[0]) - (fp[offset] + fm[offset]));

        f1 = fm;
        if (fourthdiff > maxdiff) {
          maxdiff = fourthdiff;
          bisectdim = dim;
        }
      }

      r->bisectdim = bisectdim;
    }
    team_member.team_barrier();
    for (perm = 1; perm < FEVAL / blockdim; ++perm) {
      int pIndex = perm * blockdim + threadIdx;
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 rlows,
                                                 rhighs,
                                                 global_lows,
                                                 sum,
                                                 constMem,
                                                 ranges,
                                                 jacobian,
                                                 generators,
                                                 sdata.data(),
                                                 fevals,
                                                 team_member);
    }

    // Balance permutations
    pIndex = perm * blockdim + threadIdx;
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + threadIdx;
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 rlows,
                                                 rhighs,
                                                 global_lows,
                                                 sum,
                                                 constMem,
                                                 ranges,
                                                 jacobian,
                                                 generators,
                                                 sdata.data(),
                                                 fevals,
                                                 team_member);
    }

    team_member.team_barrier();
    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum(sum[i], team_member);
    }

    if (threadIdx == 0) {

      Result* r = &region->result; // ptr to shared Mem

      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0.;

        constexpr int NSETS = 9;
        for (int s = 0; s < NSETS; ++s) {
          maxerr = max(maxerr,
                       fabs(sum[rul + 1] +
                            (constMem.GPUScale[s * NRULES + rul]) * sum[rul]) *
                         (constMem.GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = vol* sum[0];
      const T errcoeff[3] = {5., 1., 5.};
      r->err = vol * ((errcoeff[0] * sum[1] <= sum[2] &&
                          errcoeff[0] * sum[2] <= sum[3]) ?
                           errcoeff[1] * sum[1] :
                           errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));
    }
  }

  template <typename T>
  KOKKOS_INLINE_FUNCTION T
  scale_point(const T val, T low, T high)
  {
    return low + (high - low) * val;
  }

}

#endif
