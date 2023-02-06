#include "kokkos/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "catch2/catch.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class PTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    double res = 15.37 + x * 0 + y * 0;
    return res;
  }
};

class NTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    double res = -15.37 + x * 0 + y * 0;
    return res;
  }
};

class ZTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    return x * 0 + y * 0;
  }
};

TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  constexpr bool use_custom = true;
  int iteration = 0;
  bool compute_relerr_error_reduction = false;

  PTest integrand;
  PTest* d_integrand = quad::cuda_copy_to_managed<PTest>(integrand);
  quad::Volume<double, ndim> vol;
  Cubature_rules<double, ndim, use_custom> rules;
  rules.set_device_volume(vol.lows, vol.highs);
  double integral_val = 15.37;

  for (int splits_per_dim = 5; splits_per_dim < 15; splits_per_dim++) {
    Sub_regions<double, ndim> sub_regions(splits_per_dim);
    size_t nregions = sub_regions.size;
    Region_characteristics<ndim> characteristics(nregions);
    Region_estimates<double, ndim> estimates(nregions);

    auto result = rules.template apply_cubature_integration_rules<PTest>(
      d_integrand,
      iteration,
      sub_regions,
      estimates,
      characteristics,
      compute_relerr_error_reduction);

    auto h_estimates = Kokkos::create_mirror_view(estimates.integral_estimates);
    Kokkos::deep_copy(h_estimates, estimates.integral_estimates);
    double sum = 0.;
    for (size_t i = 0; i < nregions; ++i)
      sum += h_estimates[i];
    double true_val = integral_val / nregions;

    for (size_t i = 0; i < nregions; ++i) {
      CHECK(h_estimates[i] == Approx(true_val).epsilon(1.e-6));
    }
  }
}

TEST_CASE("Negative Positive Value Function")
{
  constexpr bool use_custom = true;
  constexpr int ndim = 2;
  int iteration = 0;
  bool compute_relerr_error_reduction = false;

  NTest integrand;
  NTest* d_integrand = quad::cuda_copy_to_managed(integrand);
  quad::Volume<double, ndim> vol;
  Cubature_rules<double, ndim, use_custom> rules;
  rules.set_device_volume(vol.lows, vol.highs);
  double integral_val = -15.37;

  for (int splits_per_dim = 5; splits_per_dim < 15; splits_per_dim++) {
    Sub_regions<double, ndim> sub_regions(splits_per_dim);
    size_t nregions = sub_regions.size;
    Region_characteristics<ndim> characteristics(nregions);
    Region_estimates<double, ndim> estimates(nregions);

    auto result = rules.template apply_cubature_integration_rules<NTest>(
      d_integrand,
      iteration,
      sub_regions,
      estimates,
      characteristics,
      compute_relerr_error_reduction);

    auto h_estimates = Kokkos::create_mirror_view(estimates.integral_estimates);
    Kokkos::deep_copy(h_estimates, estimates.integral_estimates);

    double sum = 0.;
    for (size_t i = 0; i < nregions; ++i)
      sum += h_estimates[i];

    double true_val = integral_val / nregions;

    for (size_t i = 0; i < nregions; ++i) {
      CHECK(h_estimates[i] == Approx(true_val).epsilon(1.e-6));
    }
  }
}

TEST_CASE("Zero Positive Value Function")
{
  constexpr bool use_custom = true;

  constexpr int ndim = 2;
  int iteration = 0;
  bool compute_relerr_error_reduction = false;

  ZTest integrand;
  ZTest* d_integrand = quad::cuda_copy_to_managed(integrand);
  quad::Volume<double, ndim> vol;
  Cubature_rules<double, ndim, use_custom> rules;
  rules.set_device_volume(vol.lows, vol.highs);
  double integral_val = 0.;

  for (int splits_per_dim = 5; splits_per_dim < 15; splits_per_dim++) {
    Sub_regions<double, ndim> sub_regions(splits_per_dim);
    size_t nregions = sub_regions.size;
    Region_characteristics<ndim> characteristics(nregions);
    Region_estimates<double, ndim> estimates(nregions);

    auto result = rules.template apply_cubature_integration_rules<ZTest>(
      d_integrand,
      iteration,
      sub_regions,
      estimates,
      characteristics,
      compute_relerr_error_reduction);

    auto h_estimates = Kokkos::create_mirror_view(estimates.integral_estimates);
    Kokkos::deep_copy(h_estimates, estimates.integral_estimates);

    double sum = 0.;
    for (size_t i = 0; i < nregions; ++i)
      sum += h_estimates[i];

    double true_val = integral_val / nregions;

    for (size_t i = 0; i < nregions; ++i) {
      CHECK(h_estimates[i] == Approx(true_val).epsilon(1.e-6));
    }
  }
}