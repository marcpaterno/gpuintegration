#ifndef CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH

#include <CL/sycl.hpp>
#include "oneAPI/pagani/quad/quad.h"
#include "common/oneAPI/Volume.dp.hpp"
#include "common/oneAPI/cudaApply.dp.hpp"
#include "common/oneAPI/cudaArray.dp.hpp"
#include "common/oneAPI/cudaUtil.h"
#include "oneAPI/pagani/quad/GPUquad/Func_Eval.hpp"

namespace quad {
  template <typename T>
  T
  Sq(T x)
  {
    return x * x;
  }

  template <typename T>
  T
  warpReduceSum(T val, sycl::nd_item<3> item_ct1)
  {
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 16);
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 8);
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 4);
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 2);
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 1);
    return val;
  }

  template <typename T>
  T
  warpReduceSum(T val, sycl::nd_item<1> item_ct1)
  {
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 16);
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 8);
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 4);
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 2);
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 1);
    return val;
  }

  template <typename T>
  T
  blockReduceSum(T val, sycl::nd_item<1> item_ct1, T* shared)
  {

    // why was this set to 8?
    int lane = item_ct1.get_local_id(0) % 32; // 32 is for warp size
    int wid = item_ct1.get_local_id(0) >> 5 /* threadIdx.x / 32  */;

    val = warpReduceSum(val, item_ct1);
    if (lane == 0) {
      shared[wid] = val;
    }

    item_ct1.barrier(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val =
      (item_ct1.get_local_id(0) < (item_ct1.get_local_range().get(0) >> 5)) ?
        shared[lane] :
        0;

    item_ct1.barrier();

    if (wid == 0)
      val = warpReduceSum(val, item_ct1); // Final reduce within first warp

    return val;
  }

  template <typename T>
  T
  blockReduceSum(T val, sycl::nd_item<3> item_ct1, T* shared)
  {

    // why was this set to 8?
    int lane = item_ct1.get_local_id(2) % 32; // 32 is for warp size
    int wid = item_ct1.get_local_id(2) >> 5 /* threadIdx.x / 32  */;

    val = warpReduceSum(val, item_ct1);
    if (lane == 0) {
      shared[wid] = val;
    }
    
    item_ct1.barrier(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val =
      (item_ct1.get_local_id(2) < (item_ct1.get_local_range().get(0) >> 5)) ?
        shared[lane] :
        0;

    item_ct1.barrier();

    if (wid == 0)
      val = warpReduceSum(val, item_ct1); // Final reduce within first warp

    return val;
  }

  template <typename T>
  T
  computeReduce(T sum, T* sdata, sycl::nd_item<3> item_ct1)
  {
    sdata[item_ct1.get_local_id(2)] = sum;

    
    item_ct1.barrier();
    // is it wise to use shlf_down_sync, sdata[BLOCK_SIZE]
    // contiguous range pattern
    for (size_t offset = item_ct1.get_local_range().get(2) / 2; offset > 0;
         offset >>= 1) {
      if (item_ct1.get_local_id(2) < offset) {
        sdata[item_ct1.get_local_id(2)] +=
          sdata[item_ct1.get_local_id(2) + offset];
      }
      
      item_ct1.barrier();
    }
    return sdata[0];
  }

  template <typename IntegT, typename T, int NDIM, int debug = 0>
  void
  computePermutation(IntegT* d_integrand,
                     int pIndex,
                     Bounds* b,
                     GlobalBounds sBound[],
                     T* sum,
                     Structures<double>& constMem,
                     T range[],
                     T* jacobian,
                     double* generators,
                     T* sdata,
                     sycl::nd_item<1> item_ct1,
                     quad::Func_Evals<NDIM>& fevals)
  {
    gpu::cudaArray<T, NDIM> x;
	
	
    #pragma unroll NDIM
	for (int dim = 0; dim < NDIM; ++dim) {
      const T generator =
        generators[CuhreFuncEvalsPerRegion<NDIM>() * dim + pIndex];
	  x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }

    const T fun = gpu::apply(*d_integrand, x) * (*jacobian);
    sdata[item_ct1.get_local_id(0)] = fun; // target for reduction
                                           
    const int gIndex = constMem._gpuGenPermGIndex[pIndex];
	
    if constexpr (debug >= 2) {
      fevals[item_ct1.get_group(0) * CuhreFuncEvalsPerRegion<NDIM>() + pIndex]
        .store(x, sBound, b);
      fevals[item_ct1.get_group(0) * CuhreFuncEvalsPerRegion<NDIM>() + pIndex]
        .store(gpu::apply(*d_integrand, x), pIndex);
    }

#pragma unroll 5
    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * constMem._cRuleWt[gIndex * NRULES + rul];
    }
  }

  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
  template <typename IntegT, typename T, int NDIM, int blockdim, int debug = 0>
  void
  SampleRegionBlock(IntegT* d_integrand,
                    Structures<double>& constMem,
                    Region<NDIM> sRegionPool[],
                    GlobalBounds sBound[],
                    T* vol,
                    int* maxdim,
                    T range[],
                    T* jacobian,
                    double* generators,
                    sycl::nd_item<1> item_ct1,
                    T* shared,
                    T* sdata,
                    quad::Func_Evals<NDIM>& fevals)
  {
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[0];
    int perm = 0;
    constexpr int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + item_ct1.get_local_id(0);
    constexpr int FEVAL = CuhreFuncEvalsPerRegion<NDIM>();
    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 sum,
                                                 constMem,
                                                 range,
                                                 jacobian,
                                                 generators,
                                                 sdata,
                                                 item_ct1,
                                                 fevals);
    }

    
    item_ct1.barrier();

    if (item_ct1.get_local_id(0) == 0) {
      const T ratio = Sq(constMem._gpuG[2 * NDIM] / constMem._gpuG[1 * NDIM]);
      T* f = &sdata[0];
      Result* r = &region->result;
      T* f1 = f;
      T base = *f1 * 2 * (1 - ratio);
      T maxdiff = 0.;
      int bisectdim = *maxdim;
      for (int dim = 0; dim < NDIM; ++dim) {
        T* fp = f1 + 1;
        T* fm = fp + 1;
        T fourthdiff = sycl::fabs(base + ratio * (fp[0] + fm[0]) -
                                  (fp[offset] + fm[offset]));

        f1 = fm;
        if (fourthdiff > maxdiff) {
          maxdiff = fourthdiff;
          bisectdim = dim;
        }
      }

      r->bisectdim = bisectdim;
    }

    item_ct1.barrier();

	#pragma unroll 1
    for (perm = 1; perm < FEVAL / blockdim; ++perm) {
      int pIndex = perm * blockdim + item_ct1.get_local_id(0);
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 sum,
                                                 constMem,
                                                 range,
                                                 jacobian,
                                                 generators,
                                                 sdata,
                                                 item_ct1,
                                                 fevals);

    }
    //__syncthreads();
    // Balance permutations
    pIndex = perm * blockdim + item_ct1.get_local_id(0);
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + item_ct1.get_local_id(0);
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                                 pIndex,
                                                 region->bounds,
                                                 sBound,
                                                 sum,
                                                 constMem,
                                                 range,
                                                 jacobian,
                                                 generators,
                                                 sdata,
                                                 item_ct1,
                                                 fevals);
    }

    item_ct1.barrier();
	#pragma unroll 5
    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum(sum[i], item_ct1, shared);
      // sum[i] = sycl::reduce_over_group(item_ct1.get_group(), sum[i],
      // sycl::plus<>());
      //__syncthreads();
    }

    if (item_ct1.get_local_id(0) == 0) {
      Result* r = &region->result;

	  #pragma unroll 3
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0.;

        constexpr int NSETS = 9;
		#pragma unroll 9
        for (int s = 0; s < NSETS; ++s) {
          maxerr =
            sycl::max(maxerr,
                sycl::fabs(sum[rul + 1] +
                     (constMem._GPUScale[s * NRULES + rul]) * sum[rul]) *
                  (constMem._GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }
	

      r->avg = (*vol) * sum[0];
      const double errcoeff[3] = {5., 1., 5.};
      r->err =
        (*vol) *
        ((errcoeff[0] * sum[1] <= sum[2] && errcoeff[0] * sum[2] <= sum[3]) ?
           errcoeff[1] * sum[1] :
           errcoeff[2] * sycl::max(sycl::max(sum[1], sum[2]), sum[3]));
	
	}
  }

}

#endif
