#ifndef WORKSPACE_CUH
#define WORKSPACE_CUH

#include <CL/sycl.hpp>
#include "common/integration_result.hh"
#include "oneAPI/pagani/quad/GPUquad/Region_estimates.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Sub_regions.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Region_characteristics.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/hybrid.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/PaganiUtils.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Sub_region_splitter.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Sub_region_filter.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/heuristic_classifier.dp.hpp"
#include "common/oneAPI/cuhreResult.dp.hpp"
#include "common/oneAPI/Volume.dp.hpp"
#include <fstream>
#include <cmath>

template <bool debug_ters = false>
void
output_iter_data()
{
  if constexpr (!debug_ters)
    return;
}

template <size_t ndim, bool use_custom = false>
class Workspace {
  using Estimates = Region_estimates<ndim>;
  using Sub_regs = Sub_regions<ndim>;
  using Regs_characteristics = Region_characteristics<ndim>;
  using Res = numint::integration_result;
  using Filter = Sub_regions_filter<ndim, use_custom>;
  using Splitter = Sub_region_splitter<ndim>;
  using Classifier = Heuristic_classifier<ndim, use_custom>;

private:
  void fix_error_budget_overflow(Region_characteristics<ndim>* classifiers,
                                 const numint::integration_result& finished,
                                 const numint::integration_result& iter,
                                 numint::integration_result& iter_finished,
                                 const double epsrel);
  bool heuristic_classify(Classifier& classifier_a,
                          Regs_characteristics& characteristics,
                          const Estimates& estimates,
                          numint::integration_result& finished,
                          const numint::integration_result& iter,
                          const numint::integration_result& cummulative);

  Cubature_rules<ndim> rules;

public:
  Workspace() = default;
  // Workspace(double* lows, double* highs):Cubature_rules<ndim>(lows, highs){}

  template <typename IntegT, bool debug = false>
  numint::integration_result integrate(const IntegT& integrand,
                                       double epsrel,
                                       double epsabs,
                                       quad::Volume<double, ndim>& vol,
                                       const std::string& optional = "default");

  template <typename IntegT,
            bool predict_split = false,
            bool collect_iters = false,
            bool collect_sub_regions = false,
            int debug = 0>
  numint::integration_result integrate(const IntegT& integrand,
                                       Sub_regions<ndim>& subregions,
                                       double epsrel,
                                       double epsabs,
                                       quad::Volume<double, ndim>& vol,
                                       bool relerr_classification = true,
                                       const std::string& optional = "default");
};

template <size_t ndim, bool use_custom>
bool
Workspace<ndim, use_custom>::heuristic_classify(
  Classifier& classifier_a,
  Region_characteristics<ndim>& characteristics,
  const Estimates& estimates,
  numint::integration_result& finished,
  const Res& iter,
  const numint::integration_result& cummulative)
{
  const double ratio =
    static_cast<double>(
      classifier_a.device_mem_required_for_full_split(characteristics.size)) /
    static_cast<double>(free_device_mem(characteristics.size, ndim));
  const bool classification_necessary = ratio > 1.;

  if (!classifier_a.classification_criteria_met(characteristics.size)) {
    const bool must_terminate = classification_necessary;
    return must_terminate;
  }

  Classification_res hs_results =
    classifier_a.classify(characteristics.active_regions,
                          estimates.error_estimates,
                          estimates.size,
                          iter.errorest,
                          finished.errorest,
                          cummulative.errorest);
  const bool hs_classify_success =
    hs_results.pass_mem && hs_results.pass_errorest_budget;

  if (hs_classify_success) {
    auto q_ct1 = sycl::queue(sycl::gpu_selector());
    sycl::free(characteristics.active_regions, q_ct1);
    characteristics.active_regions = hs_results.active_flags;
    finished.estimate = iter.estimate - dot_product<double, double, use_custom>(
                                          characteristics.active_regions,
                                          estimates.integral_estimates,
                                          characteristics.size);
    finished.errorest = hs_results.finished_errorest;
  }

  const bool must_terminate =
    (!hs_classify_success && classification_necessary) ||
    hs_results.num_active == 0;
  return must_terminate;
}

template <size_t ndim, bool use_custom>
void
Workspace<ndim, use_custom>::fix_error_budget_overflow(
  Region_characteristics<ndim>* characteristics,
  const numint::integration_result& cummulative_finished,
  const numint::integration_result& iter,
  numint::integration_result& iter_finished,
  const double epsrel)
{
  double leaves_estimate = cummulative_finished.estimate + iter.estimate;
  double leaves_finished_errorest =
    cummulative_finished.errorest + iter_finished.errorest;

  if (leaves_finished_errorest > fabs(leaves_estimate) * epsrel) {
    size_t num_threads = 256;
    double* active_regions = characteristics->active_regions;
    size_t size = characteristics->size;
    size_t num_blocks = characteristics->size / num_threads +
                        (characteristics->size % num_threads == 0 ? 0 : 1);
    auto q_ct1 = sycl::queue(sycl::gpu_selector());
    q_ct1
      .parallel_for(
        sycl::nd_range(sycl::range(num_blocks) * sycl::range(num_threads),
                       sycl::range(num_threads)),
        [=](sycl::nd_item<1> item_ct1) {
          quad::set_array_to_value<double>(active_regions, size, 1., item_ct1);
        })
      .wait();

    iter_finished.errorest = 0.;
    iter_finished.estimate = 0.;
  }
}

template <size_t ndim, bool use_custom>
template <typename IntegT,
          bool predict_split,
          bool collect_iters,
          bool collect_sub_regions,
          int debug>
numint::integration_result
Workspace<ndim, use_custom>::integrate(const IntegT& integrand,
                                       Sub_regions<ndim>& subregions,
                                       double epsrel,
                                       double epsabs,
                                       quad::Volume<double, ndim>& vol,
                                       bool relerr_classification,
                                       const std::string& optional)
{
  using CustomTimer = std::chrono::high_resolution_clock::time_point;
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  CustomTimer timer;
  auto q_ct1 = sycl::queue(sycl::gpu_selector());
  Res cummulative;
  Recorder<debug> iter_recorder;
  Recorder<debug> time_breakdown;
  rules.set_device_volume(vol.lows, vol.highs);
  Estimates prev_iter_estimates;

  Classifier classifier_a(epsrel, epsabs);
  cummulative.status = 1;
  bool compute_relerr_error_reduction = false;
  IntegT* d_integrand = quad::cuda_copy_to_managed(integrand);

  if constexpr (debug > 0) {
    time_breakdown.outfile.open("oneapi_pagani_time_breakdown.csv");
    time_breakdown.outfile << "id, ndim, epsrel, it, name, time" << std::endl;
    iter_recorder.outfile.open("oneapi_pagani_iters.csv");
    iter_recorder.outfile << "it, estimate, errorest, nregions" << std::endl;
  }

  for (size_t it = 0; it < 700 && subregions.size > 0; it++) {
    size_t num_regions = subregions.size;
    Regs_characteristics characteristics(subregions.size);
    Estimates estimates(subregions.size);

    if constexpr (debug > 0) {
      timer = std::chrono::high_resolution_clock::now();
    }

    Res iter = rules.template apply_cubature_integration_rules<IntegT,
                                                               collect_iters,
                                                               debug>(
      d_integrand,
      &subregions,
      &estimates,
      &characteristics,
      compute_relerr_error_reduction);

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "apply_cubature_rules," << dt.count() << std::endl;
    }

    if (predict_split) {
      relerr_classification =
        subregions.size <= 15000000 && it < 15 && cummulative.nregions == 0 ?
          false :
          true;
    }

    if constexpr (debug > 0) {
      timer = std::chrono::high_resolution_clock::now();
    }

    two_level_errorest_and_relerr_classify<ndim>(&estimates,
                                                 &prev_iter_estimates,
                                                 &characteristics,
                                                 epsrel,
                                                 relerr_classification);
    iter.errorest =
      reduction<double, use_custom>(estimates.error_estimates, subregions.size);
    if (predict_split) {
      if (cummulative.nregions == 0 && it == 15) {
        subregions.take_snapshot();
      }
    }

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "two_level_errorest," << dt.count() << std::endl;
      iter_recorder.outfile << it << "," << cummulative.estimate + iter.estimate
                            << "," << cummulative.errorest + iter.errorest
                            << "," << subregions.size << std::endl;
      std::cout << it << "," << cummulative.estimate + iter.estimate << ","
                << cummulative.errorest + iter.errorest << ","
                << subregions.size << std::endl;
    }

    cummulative.iters++;

    if (accuracy_reached(epsrel,
                         epsabs,
                         std::abs(cummulative.estimate + iter.estimate),
                         cummulative.errorest + iter.errorest)) {

      cummulative.estimate += iter.estimate;
      cummulative.errorest += iter.errorest;
      cummulative.status = 0;
      cummulative.nregions += subregions.size;
      d_integrand->~IntegT();
      sycl::free(d_integrand, q_ct1);
      return cummulative;
    }

    if constexpr (debug > 0) {
      timer = std::chrono::high_resolution_clock::now();
    }

    classifier_a.store_estimate(cummulative.estimate + iter.estimate);
    Res finished =
      compute_finished_estimates<ndim>(estimates, characteristics, iter);

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "compute_finished_estimates," << dt.count() << std::endl;
      timer = std::chrono::high_resolution_clock::now();
    }

    fix_error_budget_overflow(
      &characteristics, cummulative, iter, finished, epsrel);

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "fix_error_budget_overflow," << dt.count() << std::endl;
      timer = std::chrono::high_resolution_clock::now();
    }

    if (heuristic_classify(classifier_a,
                           characteristics,
                           estimates,
                           finished,
                           iter,
                           cummulative) == true) {
      cummulative.estimate += iter.estimate;
      cummulative.errorest += iter.errorest;
      cummulative.nregions += subregions.size;
      d_integrand->~IntegT();
      sycl::free(d_integrand, q_ct1);

      if constexpr (debug > 0) {
        MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
        time_breakdown.outfile
          << optional << "," << ndim << "," << epsrel << "," << it << ","
          << "heuristic_classify," << dt.count() << std::endl;
      }

      return cummulative;
    }

    cummulative.estimate += finished.estimate;
    cummulative.errorest += finished.errorest;

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "heuristic_classify," << dt.count() << std::endl;
    }

    Filter filter_obj(subregions.size);
    size_t num_active_regions = filter_obj.filter(
      &subregions, &characteristics, &estimates, &prev_iter_estimates);
    cummulative.nregions += num_regions - num_active_regions;
    subregions.size = num_active_regions;

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile << optional << "," << ndim << "," << epsrel << ","
                             << it << ","
                             << "region_filtering," << dt.count() << std::endl;
      timer = std::chrono::high_resolution_clock::now();
    }

    Splitter splitter(subregions.size);
    splitter.split(&subregions, &characteristics);
    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile << optional << "," << ndim << "," << epsrel << ","
                             << it << ","
                             << "region_splitting," << dt.count() << std::endl;
    }
  }

  d_integrand->~IntegT();
  cummulative.nregions += subregions.size;
  sycl::free(d_integrand, q_ct1);
  return cummulative;
}

template <size_t ndim, bool use_custom>
template <typename IntegT, bool debug>
numint::integration_result
Workspace<ndim, use_custom>::integrate(const IntegT& integrand,
                                       double epsrel,
                                       double epsabs,
                                       quad::Volume<double, ndim>& vol,
                                       const std::string& optional)
{
  bool relerr_classification = true;
  size_t partitions_per_axis = 2;
  if (ndim < 5)
    partitions_per_axis = 4;
  else if (ndim <= 10)
    partitions_per_axis = 2;
  else
    partitions_per_axis = 1;

  Sub_regions<ndim> sub_regions(partitions_per_axis);
  constexpr bool predict_split = false;
  constexpr bool collect_iters = false;
  constexpr bool collect_sub_regions = false;

  numint::integration_result result =
    integrate<IntegT, predict_split, collect_iters, collect_sub_regions, debug>(
      integrand,
      sub_regions,
      epsrel,
      epsabs,
      vol,
      relerr_classification,
      optional);
  return result;
}
#endif
