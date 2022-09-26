#ifndef ONE_API_CUBATURE_RULES_H
#define ONE_API_CUBATURE_RULES_H

#include "oneAPI/quad/GPUquad/Rule.h"
#include "oneAPI/quad/util/cuhreResult.h"
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/util/MemoryUtil.h"
#include "oneAPI/quad/Rule_Params.h"
#include "oneAPI/quad/Region_characteristics.h"
#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/GPUquad/Phases.h"
#include <limits>

template <size_t ndim>
class Cubature_rules {
public:
  template <typename IntegT, int warp_size>
  cuhreResult<double> apply_cubature_integration_rules(
    sycl::queue& q,
    IntegT* integrand,
    double* lows,
    double* highs,
    double& epsrel,
    double& epsabs,
    const Sub_regions<ndim>& subregions,
    Region_estimates<ndim>& estimates,
    Region_characteristics<ndim>& chars,
    bool compute_error = false);

  void ComputeGenerators(queue& q, Rule_Params<ndim>& params);

  Cubature_rules(queue& q);

  quad::Rule<double> rule;
  Rule_Params<ndim> rule_params;
};

template <size_t ndim>
Cubature_rules<ndim>::Cubature_rules(queue& q)
{

  size_t fEvalPerRegion = CuhreFuncEvalsPerRegion<ndim>();
  int key = 0;
  int verbose = 0;
  rule.Init(ndim, fEvalPerRegion, key, verbose);
  rule_params.init(q, rule);

  try {
    ComputeGenerators(q, rule_params);
  }
  catch (exception const& e) {
    std::cout << "Caught a synchronous SYCL exception:" << e.what();
    return;
  }
}

template <size_t ndim>
void
Cubature_rules<ndim>::ComputeGenerators(queue& q, Rule_Params<ndim>& params)
{

  /*
      rule params holds all the input vectors
      rule params contains the output vector "generators"
  */

  constexpr size_t block_size = 64;
  size_t feval = get_feval<ndim>();
  q.submit([&](auto& cgh) {
     cgh.parallel_for(sycl::range<1>(feval), [=](sycl::id<1> feval_id) {
       size_t tid = feval_id[0];
       double g[ndim] = {0.};

       int posCnt =
         params._gpuGenPermVarStart[tid + 1] - params._gpuGenPermVarStart[tid];
       int gIndex = params._gpuGenPermGIndex[tid];

       for (int posIter = 0; posIter < posCnt; ++posIter) {
         int pos = params._gpuGenPos[params._gpuGenPermVarStart[tid] + posIter];
         int absPos = sycl::abs(pos);

         if (pos == absPos) {
           g[absPos - 1] = params._gpuG[gIndex * ndim + posIter];

         } else {
           g[absPos - 1] = -params._gpuG[gIndex * ndim + posIter];
         }
       }

       for (size_t dim = 0; dim < ndim; dim++) {
         params._generators[feval * dim + tid] = g[dim];
       }
     });
   })
    .wait_and_throw();
}

template <size_t ndim>
template <typename IntegT, int warp_size>
cuhreResult<double>
Cubature_rules<ndim>::apply_cubature_integration_rules(
  sycl::queue& q,
  IntegT* integrand,
  double* lows,
  double* highs,
  double& epsrel,
  double& epsabs,
  const Sub_regions<ndim>& subregions,
  Region_estimates<ndim>& estimates,
  Region_characteristics<ndim>& chars,
  bool compute_error)
{

  const int nsets = 9;
  const int feval = static_cast<int>(CuhreFuncEvalsPerRegion<ndim>());

  const double val = 1.;
  constexpr int num_threads_per_work_group = 64;
  quad::parallel_fill<double>(q, chars.active_regions, subregions.size, val);
  Structures<double> params;
  params._gpuG = rule_params._gpuG;
  params._cRuleWt = rule_params._cRuleWt;
  params._GPUScale = rule_params._GPUScale;
  params._GPUNorm = rule_params._GPUNorm;
  params._gpuGenPos = rule_params._gpuGenPos;
  params._gpuGenPermGIndex = rule_params._gpuGenPermGIndex;
  params._gpuGenPermVarCount = rule_params._gpuGenPermVarCount;
  params._gpuGenPermVarStart = rule_params._gpuGenPermVarStart;

  quad::
    integrate_kernel<IntegT, ndim, num_threads_per_work_group /*, warp_size*/>(
      q,
      integrand,
      subregions.dLeftCoord,
      subregions.dLength,
      subregions.size,
      estimates.integral_estimates,
      estimates.error_estimates,
      chars.active_regions,
      chars.sub_dividing_dim,
      epsrel,
      epsabs,
      params,
      lows,
      highs,
      rule_params._generators);

  cuhreResult<double> res;
  res.estimate =
    quad::reduction<double>(q, estimates.integral_estimates, subregions.size);
  res.errorest =
    compute_error ?
      quad::reduction<double>(q, estimates.error_estimates, subregions.size) :
      std::numeric_limits<double>::infinity();
  return res;
}

#endif
