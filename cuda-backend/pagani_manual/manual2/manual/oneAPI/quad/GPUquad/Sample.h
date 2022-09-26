#ifndef ONE_API_QUAD_GPUQUAD_SAMPLE_CUH
#define ONE_API_QUAD_GPUQUAD_SAMPLE_CUH

#include <CL/sycl.hpp>

#include "oneAPI/quad/Rule_Params.h"
#include "oneAPI/quad/util/cudaArray.h"
#include "oneAPI/quad/util/cudaApply.h"
#include "oneAPI/quad/util/cudaUtil.h"
#include "oneAPI/quad/quad.h"

namespace quad {

  template <typename T>
  T
  Sq(T x)
  {
    return x * x;
  }

  template <typename T>
  using shared = sycl::
    accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>;

  template <typename T, int blockdim>
  double
  block_reduce(sycl::nd_item<1> item,
               T val,
               shared<T> fast_access_buffer,
               sycl::stream str)
  {
    const size_t work_group_id = item.get_group_linear_id();
    const size_t work_group_tid = item.get_local_id();

    auto sg = item.get_sub_group();

    int sg_id = sg.get_group_id()[0];
    int l_id = sg.get_local_id()[0];

    val = sycl::reduce_over_group(sg, val, sycl::plus<>()); // warp reduction

    item.barrier(
      sycl::access::fence_space::local_space); // consider global_and_local

    if (l_id == 0)
      fast_access_buffer[sg_id] = val;

    item.barrier(
      sycl::access::fence_space::local_space); // consider global_and_local

    val = work_group_tid < sg.get_group_range()[0] ?
            fast_access_buffer[work_group_tid] :
            0.; // only warp 0 writes to val

    item.barrier(sycl::access::fence_space::local_space);

    // posible optimization, causes issue on Iris
    // if(sg_id == 0)
    {
      val = sycl::reduce_over_group(sg, val, sycl::plus<>());
    }
    return val;
  }

  template <typename F, int ndim>
  void
  computePermutation(F* d_integrand,
                     int pIndex,
                     Bounds b[],
                     GlobalBounds sBound[],
                     double* sum,
                     const Structures<double> constMem,
                     double* range,
                     double* jacobian,
                     double* generators,
                     double* sdata,
                     sycl::nd_item<1> item)
  {
    gpu::cudaArray<double, ndim> x;
    for (size_t dim = 0; dim < ndim; ++dim) {
      const double generator =
        (generators[CuhreFuncEvalsPerRegion<ndim>() * dim + pIndex]);
      x[dim] = sBound[dim].unScaledLower + ((.5 + generator) * b[dim].lower +
                                            (.5 - generator) * b[dim].upper) *
                                             range[dim];
    }

    const double fun = gpu::apply(*d_integrand, x) * (jacobian[0]);
    sdata[item.get_local_id(0)] = fun; // target for reduction
    const int gIndex = constMem._gpuGenPermGIndex[pIndex];
#pragma unroll 5
    for (int rul = 0; rul < NRULES; ++rul) {
      sum[rul] += fun * constMem._cRuleWt[gIndex * NRULES + rul];
    }
  }

  template <typename F, int ndim, int blockdim>
  void
  sample_region_block(F* d_integrand,
                      const Structures<double> constMem,
                      Region<ndim>* sRegionPool,
                      GlobalBounds sBound[],
                      double* vol,
                      int* maxdim,
                      double* range,
                      double* jacobian,
                      double* generators,
                      sycl::nd_item<1> item,
                      double* scratch,
                      double* sdata)
  {

    Region<ndim>* const region = (Region<ndim>*)&sRegionPool[0];
    int perm = 0;
    constexpr int offset = 2 * ndim;
    double sum[NRULES] = {0.};
    Zap(sum);

    int pIndex = perm * blockdim + item.get_local_id(0);
    constexpr int feval = CuhreFuncEvalsPerRegion<ndim>();
    if (pIndex < feval) {
      computePermutation<F, ndim>(d_integrand,
                                  pIndex,
                                  region->bounds,
                                  sBound,
                                  // x,
                                  sum,
                                  constMem,
                                  range,
                                  jacobian,
                                  generators,
                                  // feval,
                                  sdata,
                                  item);
    }

    item.barrier(/*sycl::access::fence_space::local_space*/);

    if (item.get_local_id(0) == 0) {
      const double ratio =
        Sq(constMem._gpuG[2 * ndim] / constMem._gpuG[1 * ndim]);
      double* f = &sdata[0];
      Result* r = &region->result;
      double* f1 = f;
      double base = *f1 * 2 * (1 - ratio);
      double maxdiff = 0.;
      int bisectdim = maxdim[0];
      for (int dim = 0; dim < ndim; ++dim) {
        double* fp = f1 + 1;
        double* fm = fp + 1;
        double fourthdiff =
          fabs(base + ratio * (fp[0] + fm[0]) - (fp[offset] + fm[offset]));

        f1 = fm;
        if (fourthdiff > maxdiff) {
          maxdiff = fourthdiff;
          bisectdim = dim;
        }
      }

      for (int i = 0; i < 4; ++i) {
        scratch[i] = 0.;
      }

      r->bisectdim = bisectdim;
    }

    item.barrier(/*sycl::access::fence_space::local_space*/);

    for (perm = 1; perm < feval / blockdim; ++perm) {
      int pIndex = perm * blockdim + item.get_local_id(0);
      computePermutation<F, ndim>(d_integrand,
                                  pIndex,
                                  region->bounds,
                                  sBound,
                                  // x,
                                  sum,
                                  constMem,
                                  range,
                                  jacobian,
                                  generators,
                                  // feval,
                                  sdata,
                                  item);
    }

    pIndex = perm * blockdim + item.get_local_id(0);
    if (pIndex < feval) {
      computePermutation<F, ndim>(d_integrand,
                                  pIndex,
                                  region->bounds,
                                  sBound,
                                  // x,
                                  sum,
                                  constMem,
                                  range,
                                  jacobian,
                                  generators,
                                  // feval,
                                  sdata,
                                  item);
    }

    // item.barrier(sycl::access::fence_space::local_space);

    // auto wg = item.get_group();
    for (int i = 0; i < NRULES; i++) {

      // sum[i] =  block_reduce<double, blockdim>(item, sum[i], scratch, str);
      // //last one to compile and run for P630
      sum[i] = reduce_over_group(item.get_group(), sum[i], sycl::plus<>());
      // item.barrier(sycl::access::fence_space::local_space);
    }

    if (item.get_local_id(0) == 0) {
      Result* r = &region->result;

#pragma unroll 4
      for (int rul = 1; rul < NRULES - 1; ++rul) {
        double maxerr = 0;
        constexpr int nsets = 9;
#pragma unroll 9
        for (int s = 0; s < nsets; ++s) {
          maxerr = sycl::max(
            maxerr,
            sycl::fabs(sum[rul + 1] +
                       (constMem._GPUScale[s * NRULES + rul]) * sum[rul]) *
              (constMem._GPUNorm[s * NRULES + rul]));
        }
        sum[rul] = maxerr;
      }

      r->avg = (vol[0]) * sum[0];
      double errcoeff[3] = {5., 1., 5.};
      r->err =
        (vol[0]) *
        ((errcoeff[0] * sum[1] <= sum[2] && errcoeff[0] * sum[2] <= sum[3]) ?
           errcoeff[1] * sum[1] :
           errcoeff[2] * sycl::max(sycl::max(sum[1], sum[2]), sum[3]));
    }
  }

}
#endif
