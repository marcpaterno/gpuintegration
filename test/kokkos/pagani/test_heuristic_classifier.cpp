#include "externals/catch2/catch.hpp"

#include <iostream>
#include "kokkos/pagani/quad/GPUquad/heuristic_classifier.cuh"
#include "common/kokkos/cudaMemoryUtil.h"

#include <random>
#include <algorithm>

template <size_t ndim, bool use_custom>
void
emulate_estimate_collection(
  Heuristic_classifier<double, ndim, use_custom>& hs_classifier,
  size_t iters,
  double val)
{
  // we do this because iters_collected is private
  for (int i = 0; i < iters; ++i)
    hs_classifier.store_estimate(val);
}

template <size_t size>
double
sum(std::array<double, size> arr)
{
  double sum = 0.;
  for (auto i : arr)
    sum += i;
  return sum;
}

TEST_CASE("50% mem save prevented by high finished error-estimate")
{
  constexpr bool use_custom = true;
  constexpr size_t list_size = 7;
  constexpr size_t ndim = 3;
  double epsrel = 1.e-3;
  double epsabs = 1.e-12;
  Heuristic_classifier<double, ndim, use_custom> hs_classifier(epsrel, epsabs);

  size_t num_fake_iters = 20;
  double fake_estimate = 7000.;
  emulate_estimate_collection<ndim, use_custom>(
    hs_classifier, num_fake_iters, fake_estimate);
  CHECK(hs_classifier.estimate_converged() == true);

  std::array<double, list_size> estimates = {
    1000., 1000., 1000., 1000., 1000., 1000., 1000.};

  // needed error to terminate becomes 1.75, requirement to finish at least 50%
  // regions make this tough can only safely discard one or two regions before
  // hitting error-budget

  std::array<double, list_size> errors = {
    .075, .99, .079, 101.96, 101.33, 1.93, 101.99};
  ViewVectorDouble d_errors("d_errors", list_size);
  auto h_errors = Kokkos::create_mirror_view(d_errors);

  for (int i = 0; i < d_errors.extent(0); ++i)
    h_errors[i] = errors[i];
  Kokkos::deep_copy(d_errors, h_errors);

  std::array<double, list_size> active_flags = {1., 1., 1., 1., 1., 1., 1.};
  ViewVectorInt d_active_flags("d_active_flags", list_size);
  auto h_active_flags = Kokkos::create_mirror_view(d_active_flags);

  for (int i = 0; i < d_errors.extent(0); ++i)
    h_active_flags[i] = active_flags[i];
  Kokkos::deep_copy(d_active_flags, h_active_flags);

  double iter_estimate = sum<list_size>(estimates);
  double iter_errorest = sum<list_size>(errors);

  double finished_estimate = 0.;
  double finished_errorest = 4.2;

  // double iter_finished_estimate = 0.;
  double iter_finished_errorest = 0.;

  double total_estimate = iter_estimate + finished_estimate;
  double total_errorest = iter_errorest + finished_errorest;

  Classification_res results = hs_classifier.classify(d_active_flags,
                                                      d_errors,
                                                      list_size,
                                                      iter_errorest,
                                                      iter_finished_errorest,
                                                      finished_errorest);
  double percentage_mem_saved = 1. - static_cast<double>(results.num_active) /
                                       static_cast<double>(list_size);
  SECTION("Results reflected on active_regions output")
  {
    CHECK(percentage_mem_saved <= .5);
    CHECK(percentage_mem_saved > .3);
  }

  SECTION("Results and criteria captured in Classification_res object")
  {
    CHECK(results.pass_errorest_budget == true);
    CHECK(results.pass_mem == true);
    CHECK(results.max_budget_perc_to_cover > .25);
    CHECK(results.max_active_perc > .5);
  }
}