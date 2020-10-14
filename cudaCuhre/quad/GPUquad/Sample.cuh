#ifndef CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH

#include "../quad.h"
#include "../util/Volume.cuh"
#include "../util/cudaApply.cuh"
#include "../util/cudaArray.cuh"
#include "../util/cudaUtil.h"

namespace quad {

  template <typename T>
  __device__ T
  Sq(T x)
  {
    return x * x;
  }

  template <typename T>
  __device__ T
  computeReduce(T sum)
  {
    sdata[threadIdx.x] = sum;

    __syncthreads();
	//is it wise to use shlf_down_sync, sdata[BLOCK_SIZE] 
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
                     T* g,
                     gpu::cudaArray<T, NDIM>& x,
                     T* sum,
                     const Structures<T>& constMem)
  {
//this is maybe a problem, lows, highs are now unused, we rely on sBound for global bounds, and bounds b for the phase 2 starting dims, and phase 1 dims stored in dREgions, dREgionsLength
    for (int dim = 0; dim < NDIM; ++dim) {
      g[dim] = 0;
      x[dim] = 0;
    }

    int posCnt = (constMem._gpuGenPermVarStart[pIndex + 1]) -
                 (constMem._gpuGenPermVarStart[pIndex]);
    int gIndex = constMem._gpuGenPermGIndex[pIndex];
	
    for (int posIter = 0; posIter < posCnt; ++posIter) {
      int pos = (
        constMem._gpuGenPos[(constMem._gpuGenPermVarStart[pIndex]) +
                              posIter]);
      int absPos = abs(pos);
      if (pos == absPos) {
        g[absPos - 1] = (constMem._gpuG[gIndex * NDIM + posIter]);
      } else {
        g[absPos - 1] = -(constMem._gpuG[gIndex * NDIM + posIter]);
      }
    }

    T jacobian = 1;
    for (int dim = 0; dim < NDIM; ++dim) {
      x[dim] = (.5 + g[dim]) * b[dim].lower + (.5 - g[dim]) * b[dim].upper;
      T range = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;
	  jacobian = jacobian * range;
      x[dim] = sBound[dim].unScaledLower + x[dim] * range;
    }
	
    T fun = gpu::apply(*d_integrand, x);
    fun = fun * jacobian;
    sdata[threadIdx.x] = fun; // target for reduction

    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * (constMem._cRuleWt[gIndex * NRULES + rul]);
    }
  }

  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
  template <typename IntegT, typename T, int NDIM>
  __device__ void
  SampleRegionBlock(IntegT* d_integrand,
                    int sIndex,
                    const Structures<T>&  constMem,
                    int FEVAL,
                    int NSETS,
                    Region<NDIM> sRegionPool[])
  {
    __syncthreads();
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[sIndex];
    T vol = ldexp(1., -region->div); 
	
    T g[NDIM];
    gpu::cudaArray<double, NDIM> x;
    int perm = 0;

    T ratio =
      Sq((constMem._gpuG[2 * NDIM]) / (constMem._gpuG[1 * NDIM]));
    int offset = 2 * NDIM;
    int maxdim = 0;
    T maxrange = 0;

    __syncthreads();
    // set dimension range
    for (int dim = 0; dim < NDIM; ++dim) {

      Bounds* b = &region->bounds[dim];
      T range = b->upper - b->lower;
      if (range > maxrange) {
        maxrange = range;
        maxdim = dim;
      }
    }

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * BLOCK_SIZE + threadIdx.x;
    __syncthreads();
    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem);
    }

    __syncthreads();
    T* f = &sdata[0];
    __syncthreads();

    if (threadIdx.x == 0) {
      Result* r = &region->result;
      T* f1 = f;
      T base = *f1 * 2 * (1 - ratio);
      T maxdiff = 0;
      int bisectdim = maxdim;
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

    for (perm = 1; perm < FEVAL / BLOCK_SIZE; ++perm) {
      int pIndex = perm * BLOCK_SIZE + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem);
    }
    __syncthreads();
    // Balance permutations
    pIndex = perm * BLOCK_SIZE + threadIdx.x;
    if (pIndex < FEVAL) {
      int pIndex = perm * BLOCK_SIZE + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem);
    }
    __syncthreads();
    for (int i = 0; i < NRULES; ++i) {
      sum[i] = computeReduce<T>(sum[i]);
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      Result* r = &region->result;
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0;
        for (int s = 0; s < NSETS; ++s) {
          maxerr = max(maxerr,
                       fabs(sum[rul + 1] +
                            constMem._GPUScale[s * NRULES + rul] * sum[rul]) *
                         constMem._GPUNorm[s * NRULES + rul]);
        }
        sum[rul] = maxerr;
      }
	  
      r->avg = vol * sum[0];
      r->err = vol * ((errcoeff[0] * sum[1] <= sum[2] &&
                       errcoeff[0] * sum[2] <= sum[3]) ?
                        errcoeff[1] * sum[1] :
                        errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));
    }
  }
}

#endif
