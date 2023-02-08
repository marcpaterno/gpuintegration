// #define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"
#include "kokkos/pagani/quad/GPUquad/PaganiUtils.cuh"
// #include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
// #include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
// #include "common/kokkos/cudaMemoryUtil.h"
#include "common/integration_result.hh"
#include <vector>
#include <array>

TEST_CASE("Compute finished estimates")
{
  using integration_result = numint::integration_result;
  constexpr int ndim = 2;
  size_t num_regions = 100;
  Region_estimates<double, ndim> estimates(num_regions);
  Region_characteristics<ndim> characteristics(num_regions);

  double uniform_estimate = 3.2;
  double uniform_errorest = .00001;

  auto integral_estimates =
    Kokkos::create_mirror_view(estimates.integral_estimates);
  auto error_estimates = Kokkos::create_mirror_view(estimates.error_estimates);
  auto active_regions =
    Kokkos::create_mirror_view(characteristics.active_regions);

  size_t nregions = estimates.size;
  for (int i = 0; i < nregions; ++i) {
    integral_estimates[i] = uniform_estimate;
    error_estimates[i] = uniform_errorest;
  }

  Kokkos::deep_copy(estimates.integral_estimates, integral_estimates);
  Kokkos::deep_copy(estimates.error_estimates, error_estimates);

  SECTION("All finished regions")
  {
    for (int i = 0; i < nregions; ++i) {
      active_regions[i] = 0.;
    }
    Kokkos::deep_copy(characteristics.active_regions, active_regions);

    integration_result true_iter_estimate;
    true_iter_estimate.estimate =
      uniform_estimate * static_cast<double>(nregions);
    true_iter_estimate.errorest =
      uniform_errorest * static_cast<double>(nregions);

    integration_result test = compute_finished_estimates(
      estimates, characteristics, true_iter_estimate);
    CHECK(true_iter_estimate.estimate == Approx(test.estimate));
    CHECK(true_iter_estimate.errorest == Approx(test.errorest));
  }

  SECTION("Few active regions bundled together")
  {
    integration_result true_iter_estimate;
    true_iter_estimate.estimate =
      uniform_estimate * static_cast<double>(nregions);
    true_iter_estimate.errorest =
      uniform_errorest * static_cast<double>(nregions);

    double active_status = 1.;
    size_t first_index = 11; // first active region
    size_t last_index = 17;  // last active region
    double num_true_active_regions =
      static_cast<double>(last_index - first_index + 1);

    integration_result true_iter_finished_estimate;
    true_iter_finished_estimate.estimate =
      uniform_estimate * static_cast<double>(nregions) -
      uniform_estimate * num_true_active_regions;
    true_iter_finished_estimate.errorest =
      uniform_errorest * static_cast<double>(nregions) -
      uniform_errorest * num_true_active_regions;

    for (int i = 0; i < nregions; ++i) {
      bool in_active_range = i >= first_index && i <= last_index;
      active_regions[i] = in_active_range ? 1. : 0.;
    }
    Kokkos::deep_copy(characteristics.active_regions, active_regions);

    integration_result test = compute_finished_estimates(
      estimates, characteristics, true_iter_estimate);
    CHECK(test.estimate == Approx(true_iter_finished_estimate.estimate));
    CHECK(test.errorest == Approx(true_iter_finished_estimate.errorest));
  }
}
