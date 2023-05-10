#ifndef CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include "dpct-exp/cuda/pagani/quad/quad.h"
#include "dpct-exp/common/cuda/Volume.dp.hpp"
#include "dpct-exp/common/cuda/cudaApply.dp.hpp"
#include "dpct-exp/common/cuda/cudaArray.dp.hpp"
#include "dpct-exp/common/cuda/cudaUtil.h"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Func_Eval.dp.hpp"
#include <cmath>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

namespace quad {
  template <typename T>
  T
  Sq(T x)
  {
    return x * x;
  }

  template <typename T>
  T
  warpReduceSum(T val, sycl::nd_item<1> item_ct1)
  {
    /*
    DPCT1023:51: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 16);
    /*
    DPCT1023:52: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 8);
    /*
    DPCT1023:53: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 4);
    /*
    DPCT1023:54: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 2);
    /*
    DPCT1023:55: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 1);
    return val;
  }

  template <typename T>
  T
  warpReduceSum(T val, sycl::nd_item<3> item_ct1)
  {
    /*
    DPCT1023:51: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 16);
    /*
    DPCT1023:52: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 8);
    /*
    DPCT1023:53: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 4);
    /*
    DPCT1023:54: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 2);
    /*
    DPCT1023:55: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 1);
    return val;
  }

  template <typename T>
  T
  blockReduceSum(T val, sycl::nd_item<1> item_ct1, T *shared)
  {
         // why was this set to 8?
    const int lane = item_ct1.get_local_id(0) % 32; // 32 is for warp size
    const int wid = item_ct1.get_local_id(0) >> 5 /* threadIdx.x / 32  */;

    val = warpReduceSum(val, item_ct1);
    if (lane == 0) {
      shared[wid] = val;
    }
    /*
    DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // Wait for all partial reductions //I think it's safe
                        // to remove

    // read from shared memory only if that warp existed
    val =
      (item_ct1.get_local_id(0) < (item_ct1.get_local_range().get(0) >> 5)) ?
        shared[lane] :
        0;
    /*
    DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (wid == 0)
      val = warpReduceSum(val, item_ct1); // Final reduce within first warp

    return val;
  }
  
    template <typename T>
  T
  blockReduceSum(T val, sycl::nd_item<3> item_ct1, T *shared)
  {
         // why was this set to 8?
    const int lane = item_ct1.get_local_id(2) % 32; // 32 is for warp size
    const int wid = item_ct1.get_local_id(2) >> 5 /* threadIdx.x / 32  */;

    val = warpReduceSum(val, item_ct1);
    if (lane == 0) {
      shared[wid] = val;
    }
    /*
    DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // Wait for all partial reductions //I think it's safe
                        // to remove

    // read from shared memory only if that warp existed
    val =
      (item_ct1.get_local_id(2) < (item_ct1.get_local_range().get(2) >> 5)) ?
        shared[lane] :
        0;
    /*
    DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
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

    /*
    DPCT1065:58: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // is it wise to use shlf_down_sync, sdata[BLOCK_SIZE]
    // contiguous range pattern
    for (size_t offset = item_ct1.get_local_range().get(2) / 2; offset > 0;
         offset >>= 1) {
      if (item_ct1.get_local_id(2) < offset) {
        sdata[item_ct1.get_local_id(2)] +=
          sdata[item_ct1.get_local_id(2) + offset];
      }
      /*
      DPCT1065:59: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
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
                     Structures<T>& constMem,
                     T range[],
                     T* jacobian,
                     T* generators,
                     T* sdata,
                     quad::Func_Evals<NDIM>& fevals,
                     sycl::nd_item<1> item_ct1)
  {

    gpu::cudaArray<T, NDIM> x;

    // if I read shared memory in the case where we don't invoke the integrand,
    // cuda is slower than oneapi

    #pragma unroll NDIM //unroll not needed when threshold is not set
    for (int dim = 0; dim < NDIM; ++dim) {
      /*
      DPCT1026:60: The call to __ldg was removed because there is no
      correspoinding API in DPC++.
      */
      const T generator =
        generators[pagani::CuhreFuncEvalsPerRegion<NDIM>() * dim + pIndex];
      x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }

    const T fun = gpu::apply(*d_integrand, x) * (*jacobian);

    sdata[item_ct1.get_local_id(0)] = fun; // target for reduction
    /*
    DPCT1026:61: The call to __ldg was removed because there is no
    correspoinding API in DPC++.
    */
    const int gIndex = constMem.gpuGenPermGIndex[pIndex];

    if constexpr (debug >= 2) {
      // assert(fevals != nullptr);
      fevals[item_ct1.get_group(0) * pagani::CuhreFuncEvalsPerRegion<NDIM>() +
             pIndex]
        .store(x, sBound, b);
      fevals[item_ct1.get_group(0) * pagani::CuhreFuncEvalsPerRegion<NDIM>() +
             pIndex]
        .store(gpu::apply(*d_integrand, x), pIndex);
    }

    #pragma unroll NDIM //unroll not needed when threshold is not set
    for (int rul = 0; rul < NRULES; ++rul) {
      /*
      DPCT1026:62: The call to __ldg was removed because there is no
      correspoinding API in DPC++.
      */
      sum[rul] += fun * constMem.cRuleWt[gIndex * NRULES + rul];
    }
  }

  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
  template <typename IntegT, typename T, int NDIM, int blockdim, int debug = 0>
  void
  SampleRegionBlock(IntegT* d_integrand,
                    Structures<T>& constMem,
                    Region<NDIM> sRegionPool[],
                    GlobalBounds sBound[],
                    T* vol,
                    int* maxdim,
                    T range[],
                    T* jacobian,
                    T* generators,
                    quad::Func_Evals<NDIM>& fevals,
                    sycl::nd_item<1> item_ct1,
                    T *shared,
                    T *sdata)
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
    constexpr int FEVAL = pagani::CuhreFuncEvalsPerRegion<NDIM>();
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
                                                 fevals,
                                                 item_ct1);
    }

    /*
    DPCT1065:63: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (item_ct1.get_local_id(0) == 0) {
      const T ratio =
        /*
        DPCT1026:66: The call to __ldg was removed because there is no
        correspoinding API in DPC++.
        */
        Sq(constMem.gpuG[2 * NDIM] / constMem.gpuG[1 * NDIM]);
      T* f = &sdata[0];
      Result* r = &region->result;
      T* f1 = f;
      T base = *f1 * 2 * (1 - ratio);
      T maxdiff = 0;
      int bisectdim = *maxdim;
      #pragma unroll NDIM //no unroll needed when inline threshold is not set
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
    /*
    DPCT1065:64: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
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
                                                 fevals,
                                                 item_ct1);
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
                                                 fevals,
                                                 item_ct1);
    }

    /*
    DPCT1065:65: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    #pragma unroll 5
    for (int i = 0; i < NRULES; ++i) {
      sum[i] = reduce_over_group(item_ct1.get_group(), sum[i], sycl::plus<>());//blockReduceSum(sum[i], item_ct1, shared);
      //__syncthreads();
    }

    if (item_ct1.get_local_id(0) == 0) {

      Result* r = &region->result; // ptr to shared Mem

      #pragma unroll 3
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0.;

        //__ldg is missing from the loop below
        constexpr int NSETS = 9;
        #pragma unroll 9
        for (int s = 0; s < NSETS; ++s) {
          maxerr = sycl::max(
            maxerr,
            (double)(sycl::fabs(sum[rul + 1] +
                                (constMem.GPUScale[s * NRULES + rul]) *
                                  sum[rul]) *
                     (constMem.GPUNorm[s * NRULES + rul])));
        }
        sum[rul] = maxerr;
      }

      r->avg = (*vol) * sum[0];
      const T errcoeff[3] = {5., 1., 5.};
      // branching twice for each thread 0
      r->err =
        (*vol) *
        ((errcoeff[0] * sum[1] <= sum[2] && errcoeff[0] * sum[2] <= sum[3]) ?
           errcoeff[1] * sum[1] :
           errcoeff[2] * sycl::max(sycl::max(sum[1], sum[2]), sum[3]));
    }
  }

  template <typename T>
  T
  scale_point(const T val, T low, T high)
  {
    return low + (high - low) * val;
  }

  template <typename T>
  void
  rebin(T rc, int nd, T r[], T xin[], T xi[])
  {
    int i, k = 0;
    T dr = 0.0, xn = 0.0, xo = 0.0;

    // dr is the cummulative contribution

    for (i = 1; i < nd; i++) {
      // rc is the average bin contribution
      while (rc > dr) {

        dr += r[++k];
      }

      if (k > 1)
        xo = xi[k - 1];
      xn = xi[k];
      dr -= rc;

      xin[i] = xn - (xn - xo) * dr / r[k];
    }

    for (i = 1; i < nd; i++)
      xi[i] = xin[i];
    xi[nd] = 1.0;
  }
}

#endif
