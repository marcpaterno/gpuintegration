#ifndef CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH

#include "cudaPagani/quad/quad.h"
#include "cudaPagani/quad/util/Volume.cuh"
#include "cudaPagani/quad/util/cudaApply.cuh"
#include "cudaPagani/quad/util/cudaArray.cuh"
#include "cudaPagani/quad/util/cudaUtil.h"

namespace quad {

  template <typename T>
  __device__ T
  Sq(T x)
  {
    return x * x;
  }

  template <typename T>
  __device__ T
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
  __device__ T
  blockReduceSum(T val)
  {

    static __shared__ T shared[8]; // why was this set to 8?
    int lane = threadIdx.x % 32;   // 32 is for warp size
    int wid = threadIdx.x >> 5 /* threadIdx.x / 32  */;

    val = warpReduceSum(val);
    if (lane == 0) {
      shared[wid] = val;
    }
    __syncthreads(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;

    if (wid == 0)
      val = warpReduceSum(val); // Final reduce within first warp

    return val;
  }

  template <typename T>
  __device__ T
  computeReduce(T sum, T* sdata)
  {
    sdata[threadIdx.x] = sum;

    __syncthreads();
    // is it wise to use shlf_down_sync, sdata[BLOCK_SIZE]
    // contiguous range pattern
    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        sdata[threadIdx.x] += sdata[threadIdx.x + offset];
      }
      __syncthreads();
    }
    return sdata[0];
  }

  template <typename IntegT, typename T, int NDIM>
  __device__ void
  computePermutation(IntegT* d_integrand,
                     int pIndex,
                     Bounds* b,
                     GlobalBounds sBound[],
                     T* g,
                     gpu::cudaArray<T, NDIM>& x,
                     T* sum,
                     const Structures<double>& constMem,
                     T range[],
                     T* jacobian,
                     double* generators,
                     int FEVAL,
                     int iteration,
                     T* sdata)
  {

    for (int dim = 0; dim < NDIM; ++dim) {
      x[dim] = 0;
    }

    int gIndex = __ldg(&constMem._gpuGenPermGIndex[pIndex]);

    for (int dim = 0; dim < NDIM; ++dim) {
      T generator = __ldg(&generators[FEVAL * dim + pIndex]);
      x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }

    T fun = gpu::apply(*d_integrand, x) * (*jacobian);
    sdata[threadIdx.x] = fun; // target for reduction

    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * __ldg(&constMem._cRuleWt[gIndex * NRULES + rul]);
    }
  }

  template <typename IntegT, typename T, int NDIM>
  __device__ void
  verboseComputePermutation(IntegT* d_integrand,
                            int pIndex,
                            Bounds* b,
                            GlobalBounds sBound[],
                            // T* g,
                            gpu::cudaArray<T, NDIM>& x,
                            // T* sum,
                            // const Structures<double>& constMem,
                            T range[],
                            // T* jacobian,
                            double* generators,
                            int FEVAL,
                            // int iteration,
                            // T* sdata,
                            double* results,
                            double* funcEvals)
  {

    for (int dim = 0; dim < NDIM; ++dim) {
      x[dim] = 0;
    }

    // int gIndex = __ldg(&constMem._gpuGenPermGIndex[pIndex]);

    for (int dim = 0; dim < NDIM; ++dim) {
      T generator = __ldg(&generators[FEVAL * dim + pIndex]);
      x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }

    T fun = gpu::apply(*d_integrand, x) /** (*jacobian)*/;
    results[pIndex] = fun; // target for reduction

    size_t index = pIndex * NDIM;
    for (int i = 0; i < NDIM; i++) {
      funcEvals[index + i] = x[i];
    }
    // we only care about func evaluations and results
    /*for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * __ldg(&constMem._cRuleWt[gIndex * NRULES + rul]);
    }*/
  }

  template <typename IntegT, typename T, int NDIM, int blockdim>
  __device__ void
  verboseSampleRegionBlock(IntegT* d_integrand,
                           int sIndex,
                           const Structures<double>& constMem,
                           int FEVAL,
                           int NSETS,
                           Region<NDIM> sRegionPool[],
                           GlobalBounds sBound[],
                           T* vol,
                           int* maxdim,
                           T range[],
                           T* jacobian,
                           double* generators,
                           int iteration,
                           double* results,
                           double* funcEvals)
  {
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[sIndex];
    __shared__ T sdata[blockdim];
    // T g[NDIM];
    gpu::cudaArray<T, NDIM> x;
    int perm = 0;

    T ratio =
      Sq(__ldg(&constMem._gpuG[2 * NDIM]) / __ldg(&constMem._gpuG[1 * NDIM]));
    int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + threadIdx.x;

    if (pIndex < FEVAL) {
      verboseComputePermutation<IntegT, T, NDIM>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 // g,
                                                 x,
                                                 // sum,
                                                 // constMem,
                                                 range,
                                                 // jacobian,
                                                 generators,
                                                 FEVAL,
                                                 // iteration,
                                                 // sdata,
                                                 results,
                                                 funcEvals);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      T* f = &sdata[0];
      Result* r = &region->result;
      T* f1 = f;
      T base = *f1 * 2 * (1 - ratio);
      T maxdiff = 0;
      int bisectdim = *maxdim;
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
    __syncthreads();

    for (perm = 1; perm < FEVAL / blockdim; ++perm) {
      int pIndex = perm * blockdim + threadIdx.x;
      verboseComputePermutation<IntegT, T, NDIM>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 // g,
                                                 x,
                                                 // sum,
                                                 // constMem,
                                                 range,
                                                 // jacobian,
                                                 generators,
                                                 FEVAL,
                                                 // iteration,
                                                 // sdata,
                                                 results,
                                                 funcEvals);
    }
    //__syncthreads();
    // Balance permutations
    pIndex = perm * blockdim + threadIdx.x;
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + threadIdx.x;
      verboseComputePermutation<IntegT, T, NDIM>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 // g,
                                                 x,
                                                 // sum,
                                                 // constMem,
                                                 range,
                                                 // jacobian,
                                                 generators,
                                                 FEVAL,
                                                 // iteration,
                                                 // sdata,
                                                 results,
                                                 funcEvals);
    }
    // __syncthreads();

    // for (int i = 0; i < NRULES; ++i) {
    // sum[i] = blockReduceSum /*computeReduce*/ (sum[i] /*, sdata*/);
    //__syncthreads();
    //}

    /*if (threadIdx.x == 0) {
      Result* r = &region->result;
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0;
        for (int s = 0; s < NSETS; ++s) {
          maxerr =
            max(maxerr,
                fabs(sum[rul + 1] +
                     __ldg(&constMem._GPUScale[s * NRULES + rul]) * sum[rul]) *
                  __ldg(&constMem._GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = (*vol) * sum[0];
      r->err = (*vol) * ((errcoeff[0] * sum[1] <= sum[2] &&
                          errcoeff[0] * sum[2] <= sum[3]) ?
                           errcoeff[1] * sum[1] :
                           errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));
    }*/
  }

  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
  template <typename IntegT, typename T, int NDIM, int blockdim>
  __device__ void
  SampleRegionBlock(IntegT* d_integrand,
                    int sIndex,
                    const Structures<double>& constMem,
                    int FEVAL,
                    int NSETS,
                    Region<NDIM> sRegionPool[],
                    GlobalBounds sBound[],
                    T* vol,
                    int* maxdim,
                    T range[],
                    T* jacobian,
                    double* generators,
                    int iteration)
  {
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[sIndex];
    __shared__ T sdata[blockdim];
    T g[NDIM];
    gpu::cudaArray<T, NDIM> x;
    int perm = 0;

    T ratio =
      Sq(__ldg(&constMem._gpuG[2 * NDIM]) / __ldg(&constMem._gpuG[1 * NDIM]));
    int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + threadIdx.x;

    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM>(d_integrand,
                                          pIndex,
                                          region->bounds,
                                          sBound,
                                          g,
                                          x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          FEVAL,
                                          iteration,
                                          sdata);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      T* f = &sdata[0];
      Result* r = &region->result;
      T* f1 = f;
      T base = *f1 * 2 * (1 - ratio);
      T maxdiff = 0;
      int bisectdim = *maxdim;
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
    __syncthreads();

    for (perm = 1; perm < FEVAL / blockdim; ++perm) {
      int pIndex = perm * blockdim + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(d_integrand,
                                          pIndex,
                                          region->bounds,
                                          sBound,
                                          g,
                                          x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          FEVAL,
                                          iteration,
                                          sdata);
    }
    //__syncthreads();
    // Balance permutations
    pIndex = perm * blockdim + threadIdx.x;
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(d_integrand,
                                          pIndex,
                                          region->bounds,
                                          sBound,
                                          g,
                                          x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          FEVAL,
                                          iteration,
                                          sdata);
    }
    // __syncthreads();

    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum /*computeReduce*/ (sum[i] /*, sdata*/);
      //__syncthreads();
    }

    if (threadIdx.x == 0) {
      Result* r = &region->result;
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0;
        for (int s = 0; s < NSETS; ++s) {
          maxerr =
            max(maxerr,
                fabs(sum[rul + 1] +
                     __ldg(&constMem._GPUScale[s * NRULES + rul]) * sum[rul]) *
                  __ldg(&constMem._GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = (*vol) * sum[0];
      r->err = (*vol) * ((errcoeff[0] * sum[1] <= sum[2] &&
                          errcoeff[0] * sum[2] <= sum[3]) ?
                           errcoeff[1] * sum[1] :
                           errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));
    }
  }

}

#endif
