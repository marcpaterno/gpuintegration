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
                     sycl::nd_item<3> item_ct1)
  {

    gpu::cudaArray<T, NDIM> x;

    // if I read shared memory in the case where we don't invoke the integrand,
    // cuda is slower than oneapi

#pragma unroll NDIM
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

    sdata[item_ct1.get_local_id(2)] = fun; // target for reduction
    /*
    DPCT1026:61: The call to __ldg was removed because there is no
    correspoinding API in DPC++.
    */
    const int gIndex = constMem.gpuGenPermGIndex[pIndex];

    if constexpr (debug >= 2) {
      // assert(fevals != nullptr);
      fevals[item_ct1.get_group(2) * pagani::CuhreFuncEvalsPerRegion<NDIM>() +
             pIndex]
        .store(x, sBound, b);
      fevals[item_ct1.get_group(2) * pagani::CuhreFuncEvalsPerRegion<NDIM>() +
             pIndex]
        .store(gpu::apply(*d_integrand, x), pIndex);
    }

#pragma unroll 5
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
                    sycl::nd_item<3> item_ct1,
                    dpct_placeholder/*Fix the type mannually*/ *shared,
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
    int pIndex = perm * blockdim + item_ct1.get_local_id(2);
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

    if (item_ct1.get_local_id(2) == 0) {
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
      // #pragma unroll 1
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
      int pIndex = perm * blockdim + item_ct1.get_local_id(2);
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
    pIndex = perm * blockdim + item_ct1.get_local_id(2);
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + item_ct1.get_local_id(2);
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
      sum[i] = blockReduceSum(sum[i], item_ct1, shared);
      //__syncthreads();
    }

    if (item_ct1.get_local_id(2) == 0) {

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
                                __ldg(&constMem.GPUScale[s * NRULES + rul]) *
                                  sum[rul]) *
                     __ldg(&constMem.GPUNorm[s * NRULES + rul])));
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

  template <typename IntegT, typename T, int NDIM>
  void
  Vegas_assisted_computePermutation(
    IntegT* d_integrand,
    size_t num_samples,
    size_t num_passes,
    Bounds* b,
    GlobalBounds sBound[],
    T range[],
    T* jacobian,
    pagani::Curand_generator<T>& rand_num_generator,
    T& sum,
    T& sq_sum,
    T vol,
    sycl::nd_item<3> item_ct1,
    T *xi,
    T *d)
  {

    // random number generation for bin selection
    gpu::cudaArray<T, NDIM> x_random;
    constexpr size_t nbins = 100;
    int ndmx_p1 = nbins + 1;

    T dt[NDIM + 1];
    const int mxdim_p1 = NDIM + 1;
    T r[nbins + 1];
    T xin[nbins + 1];

    // size_t FEVAL = CuhreFuncEvalsPerRegion<NDIM>();
    // size_t num_passes = num_samples/FEVAL;

    if (item_ct1.get_local_id(2) == 0) {

      for (int j = 0; j <= NDIM; j++) {
        xi[j * ndmx_p1 + 1] =
          1.0; // this index is the first for each bin for each dimension

        for (int bin = 0; bin <= nbins; ++bin) {
          d[bin * (NDIM + 1) + j] = 0.;
        }
      }

      for (int bin = 1; bin <= nbins; bin++)
        r[bin] = 1.0;

      for (int dim = 1; dim <= NDIM; dim++) {
        rebin(1. / nbins, nbins, r, xin, &xi[dim * ndmx_p1]);
      }

      for (int dim = 1; dim <= NDIM; dim++) {
        for (int bin = 1; bin <= nbins; ++bin) {

          size_t xi_i = (dim) * (nbins + 1) + bin;
          size_t d_i = bin * mxdim_p1 + dim;

          if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 9)
            printf("xi %i, %i, %i, %i, %e, %e\n",
                   0,
                   item_ct1.get_group(2),
                   dim,
                   bin,
                   xi[xi_i],
                   d[d_i]);
        }
      }
    }

    /*
    DPCT1065:69: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (size_t pass = 0; pass < num_passes; ++pass) {

      T local_sq_sum = 0;
      T local_sum = 0;

      for (size_t sample = 0; sample < num_samples; ++sample) {
        // use one thread to see what contribution gets marked
        int bins[NDIM + 1];

        T wgt = 1.;

        for (int dim = 1; dim <= NDIM; ++dim) {
          // draw random number
          const T random = (rand_num_generator)();

          // select bin with that random number
          const T probability = 1. / static_cast<T>(nbins);
          const int bin = static_cast<int>(random / probability) + 1;
          bins[dim] = bin;

          const T bin_high = xi[(dim) * (nbins + 1) + bin];
          const T bin_low = xi[(dim) * (nbins + 1) + bin - 1];

          // get the true bounds
          const T region_scaled_b_high =
            scale_point(bin_high, b[dim - 1].lower, b[dim - 1].upper);
          const T region_scaled_b_low =
            scale_point(bin_low, b[dim - 1].lower, b[dim - 1].upper);
          // scale a random point at those bounds

          T rand_point_in_bin = scale_point(
            (rand_num_generator)(), region_scaled_b_low, region_scaled_b_high);
          x_random[dim - 1] = rand_point_in_bin;
          wgt *= nbins * (bin_high - bin_low);
        }

        T calls = 64. * num_passes * num_samples;
        T f = gpu::apply(*d_integrand, x_random) * (*jacobian) * wgt / calls;

        local_sum += f;
        local_sq_sum += f * f;

        for (int dim = 1; dim <= NDIM; ++dim) {
          atomicAdd(&d[bins[dim] * mxdim_p1 + dim], f * f);
        }
      }

      local_sq_sum += sqrt(local_sq_sum * num_samples);
      local_sq_sum += (local_sq_sum - local_sum) * (local_sq_sum + local_sum);

      if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0)
        printf("per-pass %e / %e / %e\n",
               local_sum,
               local_sum * local_sum,
               local_sq_sum);

      if (local_sq_sum <= 0.)
        local_sq_sum = 1.e-100;

      sum += local_sum;
      sq_sum += local_sq_sum;

      T xo = 0.;
      T xn = 0.;
      T rc = 0.;

      /*
      DPCT1065:70: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();

      if (item_ct1.get_local_id(2) == 0) {

        for (int dim = 1; dim <= NDIM; dim++) {
          for (int bin = 1; bin <= nbins; ++bin) {

            size_t xi_i = (dim) * (nbins + 1) + bin;
            size_t d_i = bin * mxdim_p1 + dim;

            if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 9)
              printf("xi %lu, %i, %i, %i, %e, %e\n",
                     pass + 1,
                     item_ct1.get_group(2),
                     dim,
                     bin,
                     xi[xi_i],
                     d[d_i]);
          }
        }

        for (int j = 1; j <= NDIM; j++) {

          // avg contribution from first two bins
          xo = d[1 * mxdim_p1 + j]; // bin 1 of dim j
          xn = d[2 * mxdim_p1 + j]; // bin 2 of dim j
          d[1 * mxdim_p1 + j] = (xo + xn) / 2.0;

          dt[j] = d[1 * mxdim_p1 + j]; // set dt sum to contribution of bin 1

          for (int i = 2; i < nbins; i++) {

            rc = xo + xn; // here xn is contr of previous bin
            xo = xn;

            xn = d[(i + 1) * mxdim_p1 + j]; // here takes contr of next bin
            d[i * mxdim_p1 + j] = (rc + xn) / 3.0; // avg of three bins
            dt[j] += d[i * mxdim_p1 + j]; // running sum of all contributions
          }

          d[nbins * mxdim_p1 + j] = (xo + xn) / 2.0; // do bin nd last
          dt[j] += d[nbins * mxdim_p1 + j];
        }

        for (int j = 1; j <= NDIM; j++) {
          if (dt[j] > 0.0) { // enter if there is any contribution only
            rc = 0.0;
            for (int i = 1; i <= nbins; i++) {
              r[i] = pow((1.0 - d[i * mxdim_p1 + j] / dt[j]) /
                           (log(dt[j]) - log(d[i * mxdim_p1 + j])),
                         .5);
              rc += r[i];
            }

            rebin(rc / nbins,
                  nbins,
                  r,
                  xin,
                  &xi[j * ndmx_p1]); // first bin of each dimension is at a diff
                                     // index
          }
        }

        for (int j = 1; j <= NDIM; j++) {
          for (int i = 1; i <= nbins; i++)
            d[i * mxdim_p1 + j] = 0.0;
        }
      }
      /*
      DPCT1065:71: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
  }

  template <typename IntegT,
            typename T,
            int NDIM,
            int blockdim,
            bool debug = false>
  void
  Vegas_assisted_SampleRegionBlock(IntegT* d_integrand,
                                   Structures<T>& constMem,
                                   Region<NDIM> sRegionPool[],
                                   GlobalBounds sBound[],
                                   T* vol,
                                   int* maxdim,
                                   T range[],
                                   T* jacobian,
                                   T* generators,
                                   quad::Func_Evals<NDIM>& fevals,
                                   unsigned int seed_init,
                                   sycl::nd_item<3> item_ct1,
                                   dpct_placeholder/*Fix the type mannually*/ *shared,
                                   T *xi,
                                   T *d,
                                   T *sdata)
  {
    pagani::Curand_generator<T> rand_num_generator(seed_init);
    Region<NDIM>* const region = (Region<NDIM>*)&sRegionPool[0];

    int perm = 0;
    constexpr int offset = 2 * NDIM;

    T sum[NRULES];
    Zap(sum);

    // Compute first set of permutation outside for loop to extract the Function
    // values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * blockdim + item_ct1.get_local_id(2);
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
    DPCT1065:72: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (item_ct1.get_local_id(2) == 0) {
      const T ratio =
        Sq(__ldg(&constMem.gpuG[2 * NDIM]) / __ldg(&constMem.gpuG[1 * NDIM]));
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
    /*
    DPCT1065:73: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (perm = 1; perm < FEVAL / blockdim; ++perm) {
      int pIndex = perm * blockdim + item_ct1.get_local_id(2);
      computePermutation<IntegT, T, NDIM>(d_integrand,
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
                                          fevals,
                                          item_ct1);
    }
    //__syncthreads();
    // Balance permutations
    pIndex = perm * blockdim + item_ct1.get_local_id(2);
    if (pIndex < FEVAL) {
      int pIndex = perm * blockdim + item_ct1.get_local_id(2);
      computePermutation<IntegT, T, NDIM>(d_integrand,
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
    // __syncthreads();

    for (int i = 0; i < NRULES; ++i) {
      sum[i] = blockReduceSum(sum[i], item_ct1, shared);
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
          maxerr =
            max(maxerr,
                fabs(sum[rul + 1] +
                     __ldg(&constMem.GPUScale[s * NRULES + rul]) * sum[rul]) *
                  __ldg(&constMem.GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = (*vol) * sum[0];

      const T errcoeff[3] = {5, 1, 5};
      r->err = (*vol) * ((errcoeff[0] * sum[1] <= sum[2] &&
                          errcoeff[0] * sum[2] <= sum[3]) ?
                           errcoeff[1] * sum[1] :
                           errcoeff[2] * max(max(sum[1], sum[2]), sum[3]));
    }
    /*
    DPCT1065:74: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    size_t num_samples = 50;
    size_t num_passes = 10;
    T ran_sum = 0.;
    T sq_sum = 0.;

    Vegas_assisted_computePermutation<IntegT, T, NDIM>(d_integrand,
                                                       num_samples,
                                                       num_passes,
                                                       region->bounds,
                                                       sBound,
                                                       range,
                                                       jacobian,
                                                       rand_num_generator,
                                                       ran_sum,
                                                       sq_sum,
                                                       vol[0],
                                                       item_ct1,
                                                       xi,
                                                       d);

    /*
    DPCT1065:75: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    ran_sum = blockReduceSum(ran_sum, item_ct1, shared);
    sq_sum = blockReduceSum(sq_sum, item_ct1, shared);
    /*
    DPCT1065:76: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (item_ct1.get_local_id(2) == 0) {
      Result* r = &region->result;
      // double mean = ran_sum / static_cast<double>(64* num_samples *
      // num_passes); double var = sq_sum / static_cast<double>(64*num_passes *
      // num_samples) - mean* mean; printf("region %i mcubes:%e +- %e (sum:%e)
      // nsamples:%i\n", blockIdx.x, vol[0]*mean, var, ran_sum,
      // num_samples*num_passes*64);

      T dxg = 1.0 / (num_passes * num_samples * 64);
      T dv2g, i;
      T calls = num_passes * num_samples * 64;
      for (dv2g = 1, i = 1; i <= NDIM; i++)
        dv2g *= dxg;
      dv2g = (calls * dv2g * calls * dv2g) / num_samples / num_samples /
             (num_samples - 1.0);

      printf("region %i pagani:%e +- %e mcubes:%e +- %e (ran_sum:%e)\n",
             item_ct1.get_group(2),
             r->avg,
             r->err,
             ran_sum,
             sqrt(sq_sum * dv2g),
             // vol[0]*mean, sqrt(sq_sum * dv2g),
             ran_sum);

      r->avg = vol[0] * ran_sum;
      r->err = sqrt(sq_sum * dv2g);
    }
  }
}

#endif
