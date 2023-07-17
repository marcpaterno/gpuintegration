#ifndef KOKKOS_WORKSPACE_CUH
#define KOKKOS_WORKSPACE_CUH

#include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_regions.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "kokkos/pagani/quad/GPUquad/hybrid.cuh"
#include "kokkos/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_region_splitter.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_region_filter.cuh"
#include "kokkos/pagani/quad/GPUquad/heuristic_classifier.cuh"
#include "common/integration_result.hh"
#include "common/kokkos/Volume.cuh"
#include "common/kokkos/cudaMemoryUtil.h"

template <bool debug_ters = false>
void
output_iter_data()
{
  if constexpr (!debug_ters)
    return;
}

template <typename T, size_t ndim, bool use_custom = false>
class Workspace {
  using Estimates = Region_estimates<T, ndim>;
  using Sub_regs = Sub_regions<T, ndim>;
  using Regs_characteristics = Region_characteristics<ndim>;
  using Filter = Sub_regions_filter<T, ndim, use_custom>;
  using Splitter = Sub_region_splitter<T, ndim>;
  using Classifier = Heuristic_classifier<T, ndim, use_custom>;
  std::ofstream outiters;

private:
  void fix_error_budget_overflow(Region_characteristics<ndim>& classifiers,
                                 const numint::integration_result& finished,
                                 const numint::integration_result& iter,
                                 numint::integration_result& iter_finished,
                                 const T epsrel);
  bool heuristic_classify(Classifier& classifier,
                          Regs_characteristics& characteristics,
                          const Estimates& estimates,
                          numint::integration_result& finished,
                          const numint::integration_result& iter,
                          const numint::integration_result& cummulative);

  Cubature_rules<T, ndim, use_custom> rules;

public:
  Workspace() = default;
  Workspace(T* lows, T* highs) : Cubature_rules<T, ndim>(lows, highs) {}
  template <typename IntegT,
            bool predict_split = false,
            bool collect_iters = false,
            int debug = 0>
  numint::integration_result integrate(const IntegT& integrand,
                                       Sub_regions<T, ndim>& subregions,
                                       T epsrel,
                                       T epsabs,
                                       quad::Volume<T, ndim> const& vol,
                                       bool relerr_classification = true,
                                       const std::string& optional = "default");

  template <typename IntegT,
            bool predict_split = false,
            bool collect_iters = false,
            int debug = 0>
  numint::integration_result integrate(const IntegT& integrand,
                                       T epsrel,
                                       T epsabs,
                                       quad::Volume<T, ndim> const& vol,
                                       bool relerr_classification = true);
};

template <typename T, size_t ndim, bool use_custom>
bool
Workspace<T, ndim, use_custom>::heuristic_classify(
  Classifier& classifier,
  Region_characteristics<ndim>& characteristics,
  const Estimates& estimates,
  numint::integration_result& finished,
  const numint::integration_result& iter,
  const numint::integration_result& cummulative)
{

  const T ratio = static_cast<T>(classifier.device_mem_required_for_full_split(
                    characteristics.size)) /
                  static_cast<T>(free_device_mem(characteristics.size, ndim));
  const bool classification_necessary = ratio > 1.;

  if (!classifier.classification_criteria_met(characteristics.size)) {
    const bool must_terminate = classification_necessary;
    return must_terminate;
  }

  Classification_res<T> hs_results =
    classifier.classify(characteristics.active_regions,
                        estimates.error_estimates,
                        estimates.size,
                        iter.errorest,
                        finished.errorest,
                        cummulative.errorest);
  const bool hs_classify_success =
    hs_results.pass_mem && hs_results.pass_errorest_budget;

  if (hs_classify_success) {
    characteristics.active_regions = hs_results.active_flags;
    finished.estimate = iter.estimate - dot_product<int, T, use_custom>(
                                          characteristics.active_regions,
                                          estimates.integral_estimates);
    finished.errorest = hs_results.finished_errorest;
  }

  const bool must_terminate =
    (!hs_classify_success && classification_necessary) ||
    hs_results.num_active == 0;
  return must_terminate;
}

template <typename T, size_t ndim, bool use_custom>
void
Workspace<T, ndim, use_custom>::fix_error_budget_overflow(
  Region_characteristics<ndim>& characteristics,
  const numint::integration_result& cummulative_finished,
  const numint::integration_result& iter,
  numint::integration_result& iter_finished,
  const T epsrel)
{

  T leaves_estimate = cummulative_finished.estimate + iter.estimate;
  T leaves_finished_errorest =
    cummulative_finished.errorest + iter_finished.errorest;

  if (leaves_finished_errorest > abs(leaves_estimate) * epsrel) {
    size_t num_threads = 256;
    size_t num_blocks = characteristics.size / num_threads +
                        (characteristics.size % num_threads == 0 ? 0 : 1);

    quad::set_array_to_value<int>(
      characteristics.active_regions.data(), characteristics.size, 1);
    iter_finished.errorest = 0.;
    iter_finished.estimate = 0.;
  }
}

template <typename T, size_t ndim, bool use_custom>
template <typename IntegT, bool predict_split, bool collect_iters, int debug>
numint::integration_result
Workspace<T, ndim, use_custom>::integrate(const IntegT& integrand,
                                          Sub_regions<T, ndim>& subregions,
                                          T epsrel,
                                          T epsabs,
                                          quad::Volume<T, ndim> const& vol,
                                          bool relerr_classification,
                                          const std::string& optional)
{
  using CustomTimer = std::chrono::high_resolution_clock::time_point;
  using MilliSeconds =
    std::chrono::duration<T, std::chrono::milliseconds::period>;

  CustomTimer timer;
  rules.set_device_volume(vol.lows, vol.highs);
  Estimates prev_iter_estimates;
  numint::integration_result cummulative;
  Recorder<debug> iter_recorder;
  Recorder<debug> time_breakdown;

  Classifier classifier(epsrel, epsabs);
  cummulative.status = 1;
  bool compute_relerr_error_reduction = false;
  IntegT* d_integrand = quad::make_gpu_integrand<IntegT>(integrand);

  if constexpr (debug > 0) {
    time_breakdown.outfile.open("kokkos_pagani_time_breakdown.csv");
    time_breakdown.outfile << "id, ndim, epsrel, it, name, time" << std::endl;
    iter_recorder.outfile.open("kokkos_pagani_iters.csv");
    iter_recorder.outfile << "it, estimate, errorest, nregions" << std::endl;
  }

  for (size_t it = 0; it < 700 && subregions.size > 0; it++) {
    size_t num_regions = subregions.size;
    Regs_characteristics characteristics(subregions.size);
    Estimates estimates(subregions.size);

    if constexpr (debug > 0) {
      timer = std::chrono::high_resolution_clock::now();
    }

    numint::integration_result iter =
      rules.template apply_cubature_integration_rules<IntegT, debug>(
        d_integrand,
        it,
        subregions,
        estimates,
        characteristics,
        compute_relerr_error_reduction);

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "apply_cubature_rules," << dt.count() << std::endl;
    }

    if constexpr (predict_split) {
      relerr_classification =
        subregions.size <= 15000000 && it < 15 && cummulative.nregions == 0 ?
          false :
          true;
    }

    if constexpr (debug > 0) {
      timer = std::chrono::high_resolution_clock::now();
    }

    two_level_errorest_and_relerr_classify<T, ndim>(estimates,
                                                    prev_iter_estimates,
                                                    characteristics,
                                                    epsrel,
                                                    relerr_classification);

    iter.errorest =
      reduction<T, use_custom>(estimates.error_estimates, subregions.size);

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

    if constexpr (predict_split) {
      if (cummulative.nregions == 0 && it == 15) {
        subregions.take_snapshot();
      }
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
      Kokkos::kokkos_free(d_integrand);
      return cummulative;
    }

    if constexpr (debug > 0) {
      timer = std::chrono::high_resolution_clock::now();
    }

    classifier.store_estimate(cummulative.estimate + iter.estimate);
    numint::integration_result finished =
      compute_finished_estimates<T, ndim, use_custom>(
        estimates, characteristics, iter);

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "compute_finished_estimates," << dt.count() << std::endl;
      timer = std::chrono::high_resolution_clock::now();
    }

    fix_error_budget_overflow(
      characteristics, cummulative, iter, finished, epsrel);

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "fix_error_budget_overflow," << dt.count() << std::endl;
      timer = std::chrono::high_resolution_clock::now();
    }

    if (heuristic_classify(classifier,
                           characteristics,
                           estimates,
                           finished,
                           iter,
                           cummulative) == true) {
      cummulative.estimate += iter.estimate;
      cummulative.errorest += iter.errorest;
      cummulative.nregions += subregions.size;
      Kokkos::kokkos_free(d_integrand);
      if constexpr (debug > 0) {
        MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
        time_breakdown.outfile
          << optional << "," << ndim << "," << epsrel << "," << it << ","
          << "heuristic_classify," << dt.count() << std::endl;
      }
      return cummulative;
    }

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "heuristic_classify," << dt.count() << std::endl;
    }

    cummulative.estimate += finished.estimate;
    cummulative.errorest += finished.errorest;

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile
        << optional << "," << ndim << "," << epsrel << "," << it << ","
        << "heuristic_classify," << dt.count() << std::endl;
      timer = std::chrono::high_resolution_clock::now();
    }

    Filter filter_obj(subregions.size);
    size_t num_active_regions = filter_obj.filter(
      subregions, characteristics, estimates, prev_iter_estimates);

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
    splitter.split(subregions, characteristics);

    if constexpr (debug > 0) {
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - timer;
      time_breakdown.outfile << optional << "," << ndim << "," << epsrel << ","
                             << it << ","
                             << "region_splitting," << dt.count() << std::endl;
    }
  }
  cummulative.nregions += subregions.size;
  Kokkos::kokkos_free(d_integrand);
  return cummulative;
}

template <typename T, size_t ndim, bool use_custom>
template <typename IntegT, bool predict_split, bool collect_iters, int debug>
numint::integration_result
Workspace<T, ndim, use_custom>::integrate(const IntegT& integrand,
                                          T epsrel,
                                          T epsabs,
                                          quad::Volume<T, ndim> const& vol,
                                          bool relerr_classification)
{
  using MilliSeconds =
    std::chrono::duration<T, std::chrono::milliseconds::period>;
  rules.set_device_volume(vol.lows, vol.highs);
  Estimates prev_iter_estimates;
  numint::integration_result cummulative;
  Recorder<debug> iter_recorder("cuda_iters.csv");

  size_t partitions_per_axis = 2;
  if (ndim < 5)
    partitions_per_axis = 4;
  else if (ndim <= 10)
    partitions_per_axis = 2;
  else
    partitions_per_axis = 1;

  Sub_regions<T, ndim> subregions(partitions_per_axis);

  Classifier classifier(epsrel, epsabs);
  cummulative.status = 1;
  bool compute_relerr_error_reduction = false;

  IntegT* d_integrand = quad::make_gpu_integrand<IntegT>(integrand);

  if constexpr (debug > 0) {

    iter_recorder.outfile << "it, estimate, errorest, nregions" << std::endl;
  }

  for (size_t it = 0; it < 700 && subregions.size > 0; it++) {
    size_t num_regions = subregions.size;
    Regs_characteristics characteristics(subregions.size);
    Estimates estimates(subregions.size);

    auto const t0 = std::chrono::high_resolution_clock::now();
    numint::integration_result iter =
      rules.template apply_cubature_integration_rules<IntegT, debug>(
        d_integrand,
        it,
        subregions,
        estimates,
        characteristics,
        compute_relerr_error_reduction);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

    if constexpr (predict_split) {
      relerr_classification =
        subregions.size <= 15000000 && it < 15 && cummulative.nregions == 0 ?
          false :
          true;
    }

    two_level_errorest_and_relerr_classify<T, ndim>(estimates,
                                                    prev_iter_estimates,
                                                    characteristics,
                                                    epsrel,
                                                    relerr_classification);
    iter.errorest =
      reduction<T, use_custom>(estimates.error_estimates, subregions.size);

    if constexpr (debug > 0)
      iter_recorder.outfile << it << "," << cummulative.estimate + iter.estimate
                            << "," << cummulative.errorest + iter.errorest
                            << "," << subregions.size << std::endl;

    if constexpr (predict_split) {
      if (cummulative.nregions == 0 && it == 15) {
        subregions.take_snapshot();
      }
    }

    if (accuracy_reached(epsrel,
                         epsabs,
                         std::abs(cummulative.estimate + iter.estimate),
                         cummulative.errorest + iter.errorest)) {
      cummulative.estimate += iter.estimate;
      cummulative.errorest += iter.errorest;
      cummulative.status = 0;
      cummulative.nregions += subregions.size;
      Kokkos::kokkos_free(d_integrand);
      return cummulative;
    }

    classifier.store_estimate(cummulative.estimate + iter.estimate);
    numint::integration_result finished =
      compute_finished_estimates<T, ndim, use_custom>(
        estimates, characteristics, iter);
    fix_error_budget_overflow(
      characteristics, cummulative, iter, finished, epsrel);
    if (heuristic_classify(classifier,
                           characteristics,
                           estimates,
                           finished,
                           iter,
                           cummulative) == true) {
      cummulative.estimate += iter.estimate;
      cummulative.errorest += iter.errorest;
      cummulative.nregions += subregions.size;
      Kokkos::kokkos_free(d_integrand);
      return cummulative;
    }

    cummulative.estimate += finished.estimate;
    cummulative.errorest += finished.errorest;
    Filter filter_obj(subregions.size);
    size_t num_active_regions = filter_obj.filter(
      subregions, characteristics, estimates, prev_iter_estimates);
    cummulative.nregions += num_regions - num_active_regions;
    subregions.size = num_active_regions;
    Splitter splitter(subregions.size);
    splitter.split(subregions, characteristics);
    cummulative.iters++;
  }
  cummulative.nregions += subregions.size;
  Kokkos::kokkos_free(d_integrand);
  return cummulative;
}

#endif
