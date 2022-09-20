#ifndef CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_SAMPLE_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/pagani/quad/quad.h"
#include "oneAPI/pagani/quad/util/Volume.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaApply.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaArray.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaUtil.h"
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
    /*
    DPCT1023:8: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 16);
    /*
    DPCT1023:9: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 8);
    /*
    DPCT1023:10: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 4);
    /*
    DPCT1023:11: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 2);
    /*
    DPCT1023:12: The DPC++ sub-group does not support mask options for
    sycl::shift_group_left.
    */
    val += sycl::shift_group_left(item_ct1.get_sub_group(), val, 1);
    return val;
  }

  template <typename T>
  T
  blockReduceSum(T val, sycl::nd_item<3> item_ct1, T *shared)
  {

     // why was this set to 8?
    int lane = item_ct1.get_local_id(2) % 32; // 32 is for warp size
    int wid = item_ct1.get_local_id(2) >> 5 /* threadIdx.x / 32  */;

    val = warpReduceSum(val, item_ct1);
    if (lane == 0) {
      shared[wid] = val;
    }
    /*
    DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val =
      (item_ct1.get_local_id(2) < (item_ct1.get_local_range().get(2) >> 5)) ?
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

    /*
    DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
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
      DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
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
                     //T* g,
                     //gpu::cudaArray<T, NDIM>& x,
                     T* sum,
                     Structures<double> constMem,
                     T range[],
                     T* jacobian,
                     double* generators,
                     //int FEVAL,
                     T* sdata,
                     sycl::nd_item<3> item_ct1,
                     quad::Func_Evals<NDIM>& fevals)
  {
        
    gpu::cudaArray<T, NDIM> x;
    for (int dim = 0; dim < NDIM; ++dim) {
      /*
      DPCT1026:18: The call to __ldg was removed because there is no
      correspoinding API in DPC++.
      */
      const T generator =
        generators[CuhreFuncEvalsPerRegion<NDIM>() * dim + pIndex];
      x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }
	
    const T fun = gpu::apply(*d_integrand, x) * (*jacobian);
    sdata[item_ct1.get_local_id(2)] = fun; // target for reduction
        /*
        DPCT1026:16: The call to __ldg was removed because there is no
        correspoinding API in DPC++.
        */
        const int gIndex = constMem._gpuGenPermGIndex[pIndex];
      
    if constexpr(debug >= 2){
        fevals[item_ct1.get_group(2) * CuhreFuncEvalsPerRegion<NDIM>() + pIndex].store(x, sBound, b);
        fevals[item_ct1.get_group(2) * CuhreFuncEvalsPerRegion<NDIM>() + pIndex].store(gpu::apply(*d_integrand, x), pIndex);
    }
      
      
      
#pragma unroll 5
    for (int rul = 0; rul < NRULES; ++rul) {
      /*
      DPCT1026:17: The call to __ldg was removed because there is no
      correspoinding API in DPC++.
      */
      sum[rul] += fun * constMem._cRuleWt[gIndex * NRULES + rul];
    }
  }

  // BLOCK SIZE has to be atleast 4*DIM+1 for the first IF
  template <typename IntegT, typename T, int NDIM, int blockdim, int debug = 0>
  void
  SampleRegionBlock(IntegT* d_integrand,
                    //int sIndex,
                     Structures<double> constMem,
                    //int FEVAL,
                    //int NSETS,
                    Region<NDIM> sRegionPool[],
                    GlobalBounds sBound[],
                    T* vol,
                    int* maxdim,
                    T range[],
                    T* jacobian,
                    double* generators,
                    sycl::nd_item<3> item_ct1,
                    T *shared,
                    T *sdata,
                    quad::Func_Evals<NDIM>& fevals)
  {
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[0];

    //T g[NDIM];
    //gpu::cudaArray<T, NDIM> x;
    int perm = 0;
    constexpr int offset = 2 * NDIM;
    
    T sum[NRULES];
    Zap(sum);
    
    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + item_ct1.get_local_id(2);
    constexpr int FEVAL = CuhreFuncEvalsPerRegion<NDIM>();
    if (pIndex < FEVAL) {
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                          pIndex,
                                          region->bounds,
                                          sBound,
                                          // g,
                                          // x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          // FEVAL,
                                          sdata,
                                          item_ct1,
                                          fevals);
    }

    /*
    DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (item_ct1.get_local_id(2) == 0) {
          /*
          DPCT1026:25: The call to __ldg was removed because there is no
          correspoinding API in DPC++.
          */
      const T ratio = Sq(constMem._gpuG[2 * NDIM] / constMem._gpuG[1 * NDIM]);
      T* f = &sdata[0];
      Result* r = &region->result;
      T* f1 = f;
      T base = *f1 * 2 * (1 - ratio);
      T maxdiff = 0;
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
    /*
    DPCT1065:24: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (perm = 1; perm < FEVAL / blockdim; ++perm) {
      int pIndex = perm * blockdim + item_ct1.get_local_id(2);
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                          pIndex,
                                          region->bounds,
                                          sBound,
                                          // g,
                                          // x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          // FEVAL,
                                          sdata,
                                          item_ct1, 
                                          fevals);
    }
    //__syncthreads();
    // Balance permutations
    pIndex = perm * blockdim + item_ct1.get_local_id(2);
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + item_ct1.get_local_id(2);
      computePermutation<IntegT, T, NDIM, debug>(d_integrand,
                                          pIndex,
                                          region->bounds,
                                          sBound,
                                          // g,
                                          // x,
                                          sum,
                                          constMem,
                                          range,
                                          jacobian,
                                          generators,
                                          // FEVAL,
                                          sdata,
                                          item_ct1,
                                          fevals);
    }
    // __syncthreads();
	
    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum(sum[i], item_ct1, shared);
	  //sum[i] = sycl::reduce_over_group(item_ct1.get_group(), sum[i], sycl::plus<>());
      //__syncthreads();
    }

    if (item_ct1.get_local_id(2) == 0) {
      Result* r = &region->result;
	  
	  #pragma unroll 4
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        T maxerr = 0;
		
		constexpr int NSETS = 9;
		#pragma unroll 9
        for (int s = 0; s < NSETS; ++s) {
          maxerr = sycl::max(
            maxerr,
            (double)(sycl::fabs(sum[rul + 1] +
                                constMem._GPUScale[s * NRULES + rul] *
                                  sum[rul]) *
                     constMem._GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }
      
      r->avg = (*vol) * sum[0];
	  const double errcoeff[3] = {5, 1, 5};
      r->err =
        (*vol) *
        ((errcoeff[0] * sum[1] <= sum[2] && errcoeff[0] * sum[2] <= sum[3]) ?
           errcoeff[1] * sum[1] :
           errcoeff[2] * sycl::max(sycl::max(sum[1], sum[2]), sum[3]));
    }
  }

}

#endif
