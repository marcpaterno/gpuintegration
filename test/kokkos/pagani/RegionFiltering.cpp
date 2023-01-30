#include "kokkos/pagani/quad/GPUquad/Sub_region_filter.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
#include "catch2/catch.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <array>

bool
regions_same(double* listA,
             double* listB,
             size_t indexA,
             size_t indexB,
             int ndim,
             size_t nregionsA,
             size_t nregionsB)
{

  for (int dim = 0; dim < ndim; ++dim) {
    bool diff = listA[indexA + dim * nregionsA] !=
                Approx(listB[indexB + dim * nregionsB]);
    // printf("dim:%i %f, %f: bool:%i\n", dim, listA[index + dim * nregions],
    // listB[index + dim * nregions], diff);
    if (diff)
      return false;
  }
  return true;
}

TEST_CASE("get_num_active_regions: 100-region-list")
{
  constexpr int ndim = 2;
  constexpr size_t num_regions = 100;
  Region_characteristics<ndim> regions(num_regions);
  Sub_regions_filter<double, ndim> filter(num_regions);

  auto host_active_regions = Kokkos::create_mirror_view(regions.active_regions);

  SECTION("All active regions")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 1;

    Kokkos::deep_copy(regions.active_regions, host_active_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == num_regions);
  }

  SECTION("All but last region is active")
  {
    host_active_regions[regions.size - 1] = 1;

    Kokkos::deep_copy(regions.active_regions, host_active_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 1);
  }

  SECTION("All but the first region is active")
  {
    host_active_regions[0] = 1;

    Kokkos::deep_copy(regions.active_regions, host_active_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 1);
  }

  SECTION("Zero active regions")
  {
    Kokkos::deep_copy(regions.active_regions, host_active_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 0);
  }

  SECTION("Only first region is inactive")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 1;
    host_active_regions[0] = 0;

    Kokkos::deep_copy(regions.active_regions, host_active_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 99);
  }

  SECTION("Only last region is inactive")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 1;
    host_active_regions[regions.size - 1] = 0;

    Kokkos::deep_copy(regions.active_regions, host_active_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 99);
  }
};

TEST_CASE("get_num_active_regions: one-region-list")
{
  constexpr int ndim = 2;
  constexpr size_t num_regions = 1;
  Region_characteristics<ndim> regions(num_regions);
  Sub_regions_filter<double, ndim> filter(num_regions);

  auto host_active_regions = Kokkos::create_mirror_view(regions.active_regions);

  SECTION("one active region")
  {
    host_active_regions[0] = 1;

    Kokkos::deep_copy(regions.active_regions, host_active_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 1);
  }

  SECTION("zero active regions")
  {
    host_active_regions[0] = 0;

    Kokkos::deep_copy(regions.active_regions, host_active_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 0);
  }
};

TEST_CASE("filter: all regions are active")
{
  constexpr int ndim = 2;
  Sub_regions<double, ndim> coordinates(10);
  size_t n = coordinates.size;
  Region_characteristics<ndim> classifications(n);
  Sub_regions_filter<double, ndim> filter(n);
  Region_estimates<double, ndim> estimates(n);
  Region_estimates<double, ndim> parents(n / 2);

  auto integrals_mirror =
    Kokkos::create_mirror_view(estimates.integral_estimates);
  auto error_mirror = Kokkos::create_mirror_view(estimates.error_estimates);
  auto parent_integrals_mirror =
    Kokkos::create_mirror_view(parents.integral_estimates);
  auto parent_error_mirror =
    Kokkos::create_mirror_view(parents.error_estimates);
  auto sub_dividing_dim =
    Kokkos::create_mirror_view(classifications.sub_dividing_dim);
  auto host_active_regions =
    Kokkos::create_mirror_view(classifications.active_regions);
  auto original_LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);
  auto original_Length = Kokkos::create_mirror_view(coordinates.dLength);

  Kokkos::deep_copy(original_LeftCoord, coordinates.dLeftCoord);
  Kokkos::deep_copy(original_Length, coordinates.dLength);

  // set region variabels to fit test scenario
  for (size_t i = 0; i < n; ++i) {
    integrals_mirror[i] = 1000.;
    error_mirror[i] = 10.;
    host_active_regions[i] = 1;
    sub_dividing_dim[i] = 0;

    if (static_cast<size_t>(i) < n / 2) {
      parent_integrals_mirror[i] = 1500.;
      parent_error_mirror[i] = 20.;
    }
  }

  Kokkos::deep_copy(classifications.active_regions, host_active_regions);
  Kokkos::deep_copy(estimates.integral_estimates, integrals_mirror);
  Kokkos::deep_copy(estimates.error_estimates, error_mirror);
  Kokkos::deep_copy(parents.integral_estimates, parent_integrals_mirror);
  Kokkos::deep_copy(parents.error_estimates, parent_error_mirror);
  Kokkos::deep_copy(sub_dividing_dim, classifications.sub_dividing_dim);

  size_t num_active =
    filter.filter(coordinates, classifications, estimates, parents);
  CHECK(num_active == n);

  // after filter, the pre-filtering integral and error-estimates become the
  // parents
  auto filtered_integrals =
    Kokkos::create_mirror_view(parents.integral_estimates);
  auto filtered_errors = Kokkos::create_mirror_view(parents.error_estimates);

  auto new_subdiv_dim =
    Kokkos::create_mirror_view(classifications.sub_dividing_dim);
  auto LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);
  auto Length = Kokkos::create_mirror_view(coordinates.dLength);

  Kokkos::deep_copy(LeftCoord, coordinates.dLeftCoord);
  Kokkos::deep_copy(Length, coordinates.dLength);
  Kokkos::deep_copy(filtered_integrals, parents.integral_estimates);
  Kokkos::deep_copy(filtered_errors, parents.error_estimates);
  Kokkos::deep_copy(new_subdiv_dim, classifications.sub_dividing_dim);

  for (size_t region = 0; region < n; ++region) {
    CHECK(filtered_integrals[region] == integrals_mirror[region]);
    CHECK(filtered_errors[region] == error_mirror[region]);
    CHECK(new_subdiv_dim[region] == sub_dividing_dim[region]);

    for (int dim = 0; dim < ndim; ++dim) {
      size_t i = region + dim * n;
      CHECK(LeftCoord[i] == original_LeftCoord[i]);
      CHECK(Length[i] == original_Length[i]);
    }
  }

  CHECK(filtered_integrals[16] == 1000.);
  CHECK(filtered_integrals[15] == 1000.);
  CHECK(filtered_integrals[14] == 1000.);

  CHECK(filtered_errors[16] == 10.);
  CHECK(filtered_errors[15] == 10.);
  CHECK(filtered_errors[14] == 10.);

  CHECK(new_subdiv_dim[16] == 0);
  CHECK(new_subdiv_dim[15] == 0);
  CHECK(new_subdiv_dim[14] == 0);

  for (int i = 0; i < n; ++i)
    CHECK(regions_same(
      LeftCoord.data(), original_LeftCoord.data(), i, i, ndim, num_active, n));
}

TEST_CASE("filter: region 15 is inactive")
{
  constexpr int ndim = 2;
  Sub_regions<double, ndim> coordinates(10);
  size_t n = coordinates.size;
  Region_characteristics<ndim> classifications(n);
  Sub_regions_filter<double, ndim> filter(n);
  Region_estimates<double, ndim> estimates(n);
  Region_estimates<double, ndim> parents(n / 2);

  auto integrals_mirror =
    Kokkos::create_mirror_view(estimates.integral_estimates);
  auto error_mirror = Kokkos::create_mirror_view(estimates.error_estimates);
  auto parent_integrals_mirror =
    Kokkos::create_mirror_view(parents.integral_estimates);
  auto parent_error_mirror =
    Kokkos::create_mirror_view(parents.error_estimates);
  auto sub_dividing_dim =
    Kokkos::create_mirror_view(classifications.sub_dividing_dim);
  auto host_active_regions =
    Kokkos::create_mirror_view(classifications.active_regions);
  auto original_LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);
  auto original_Length = Kokkos::create_mirror_view(coordinates.dLength);

  Kokkos::deep_copy(original_LeftCoord, coordinates.dLeftCoord);
  Kokkos::deep_copy(original_Length, coordinates.dLength);

  for (size_t i = 0; i < n; ++i) {
    integrals_mirror[i] = 1000.;
    error_mirror[i] = 10.;
    host_active_regions[i] = 1;
    sub_dividing_dim[i] = 0; // active regions all have 0 as sub-dividing dim

    if (i < n / 2) {
      parent_integrals_mirror[i] = 1500.;
      parent_error_mirror[i] = 20.;
    }
  }

  // give values for inactive region(s)
  integrals_mirror[15] = 999.;
  error_mirror[15] = 1.;
  host_active_regions[15] = 0;
  sub_dividing_dim[15] = 1; // active region has 1 as sub-dividing dim

  Kokkos::deep_copy(classifications.active_regions, host_active_regions);
  Kokkos::deep_copy(estimates.integral_estimates, integrals_mirror);
  Kokkos::deep_copy(estimates.error_estimates, error_mirror);
  Kokkos::deep_copy(parents.integral_estimates, parent_integrals_mirror);
  Kokkos::deep_copy(parents.error_estimates, parent_error_mirror);
  Kokkos::deep_copy(sub_dividing_dim, classifications.sub_dividing_dim);

  size_t num_active =
    filter.filter(coordinates, classifications, estimates, parents);
  CHECK(num_active == n - 1);
  // after filter, the pre-filtering integral and error-estimates become the
  // parents
  auto filtered_integrals =
    Kokkos::create_mirror_view(parents.integral_estimates);
  auto filtered_errors = Kokkos::create_mirror_view(parents.error_estimates);

  auto new_subdiv_dim =
    Kokkos::create_mirror_view(classifications.sub_dividing_dim);
  auto LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);
  auto Length = Kokkos::create_mirror_view(coordinates.dLength);

  Kokkos::deep_copy(LeftCoord, coordinates.dLeftCoord);
  Kokkos::deep_copy(Length, coordinates.dLength);
  Kokkos::deep_copy(filtered_integrals, parents.integral_estimates);
  Kokkos::deep_copy(filtered_errors, parents.error_estimates);
  Kokkos::deep_copy(new_subdiv_dim, classifications.sub_dividing_dim);

  // check index the index inactive region to see if filtered out, check
  // surrounding indices to detect any errors
  CHECK(filtered_integrals[16] == 1000.);
  CHECK(filtered_integrals[15] == 1000.);
  CHECK(filtered_integrals[14] == 1000.);

  CHECK(filtered_errors[16] == 10.);
  CHECK(filtered_errors[15] == 10.);
  CHECK(filtered_errors[14] == 10.);

  CHECK(new_subdiv_dim[16] == 0);
  CHECK(new_subdiv_dim[15] == 0);
  CHECK(new_subdiv_dim[14] == 0);

  // check regions 0, 1 for being the same and region 15 for being diffrent
  CHECK(regions_same(
    LeftCoord.data(), original_LeftCoord.data(), 0, 0, ndim, num_active, n));
  CHECK(regions_same(
    LeftCoord.data(), original_LeftCoord.data(), 1, 1, ndim, num_active, n));
  CHECK(regions_same(
    LeftCoord.data(), original_LeftCoord.data(), 14, 14, ndim, num_active, n));
  CHECK(!regions_same(
    LeftCoord.data(), original_LeftCoord.data(), 15, 15, ndim, num_active, n));
  CHECK(regions_same(
    LeftCoord.data(), original_LeftCoord.data(), 15, 16, ndim, num_active, n));
}

TEST_CASE("filter: first and last regions are inactive")
{
  constexpr int ndim = 2;
  Sub_regions<double, ndim> coordinates(10);
  size_t n = coordinates.size;
  Region_characteristics<ndim> classifications(n);
  Sub_regions_filter<double, ndim> filter(n);
  Region_estimates<double, ndim> estimates(n);
  Region_estimates<double, ndim> parents(n / 2);

  auto integrals_mirror =
    Kokkos::create_mirror_view(estimates.integral_estimates);
  auto error_mirror = Kokkos::create_mirror_view(estimates.error_estimates);
  auto parent_integrals_mirror =
    Kokkos::create_mirror_view(parents.integral_estimates);
  auto parent_error_mirror =
    Kokkos::create_mirror_view(parents.error_estimates);
  auto sub_dividing_dim =
    Kokkos::create_mirror_view(classifications.sub_dividing_dim);
  auto host_active_regions =
    Kokkos::create_mirror_view(classifications.active_regions);
  auto original_LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);
  auto original_Length = Kokkos::create_mirror_view(coordinates.dLength);

  Kokkos::deep_copy(original_LeftCoord, coordinates.dLeftCoord);
  Kokkos::deep_copy(original_Length, coordinates.dLength);

  for (size_t i = 0; i < n; ++i) {
    integrals_mirror[i] = 1000.;
    error_mirror[i] = 10.;
    host_active_regions[i] = 1;
    sub_dividing_dim[i] = 0; // active regions all have 0 as sub-dividing dim

    if (i < n / 2) {
      parent_integrals_mirror[i] = 1500.;
      parent_error_mirror[i] = 20.;
    }
  }

  // give values for inactive region 0
  integrals_mirror[0] = 999.;
  error_mirror[0] = 1.;
  host_active_regions[0] = 0;
  sub_dividing_dim[0] = 1; // active region has 1 as sub-dividing dim

  // give values for last inactive region
  integrals_mirror[n - 1] = 999.;
  error_mirror[n - 1] = 1.;
  host_active_regions[n - 1] = 0;
  sub_dividing_dim[n - 1] = 1; // active region has 1 as sub-dividing dim

  Kokkos::deep_copy(classifications.active_regions, host_active_regions);
  Kokkos::deep_copy(estimates.integral_estimates, integrals_mirror);
  Kokkos::deep_copy(estimates.error_estimates, error_mirror);
  Kokkos::deep_copy(parents.integral_estimates, parent_integrals_mirror);
  Kokkos::deep_copy(parents.error_estimates, parent_error_mirror);
  Kokkos::deep_copy(sub_dividing_dim, classifications.sub_dividing_dim);

  size_t num_active =
    filter.filter(coordinates, classifications, estimates, parents);
  CHECK(num_active == n - 2);
  // after filter, the pre-filtering integral and error-estimates become the
  // parents
  auto filtered_integrals =
    Kokkos::create_mirror_view(parents.integral_estimates);
  auto filtered_errors = Kokkos::create_mirror_view(parents.error_estimates);

  auto new_subdiv_dim =
    Kokkos::create_mirror_view(classifications.sub_dividing_dim);
  auto LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);
  auto Length = Kokkos::create_mirror_view(coordinates.dLength);

  Kokkos::deep_copy(LeftCoord, coordinates.dLeftCoord);
  Kokkos::deep_copy(Length, coordinates.dLength);
  Kokkos::deep_copy(filtered_integrals, parents.integral_estimates);
  Kokkos::deep_copy(filtered_errors, parents.error_estimates);
  Kokkos::deep_copy(new_subdiv_dim, classifications.sub_dividing_dim);

  // check index the index inactive region to see if filtered out, check
  // surrounding indices to detect any errors for(int i =0; i < n; ++i)
  //	printf("region %i %f +- %f\n", i, filtered_integrals[i],
  //filtered_errors[i]);

  CHECK(filtered_integrals[0] == 1000.);
  CHECK(filtered_integrals[1] == 1000.);
  CHECK(filtered_integrals[num_active - 2] == 1000.);
  CHECK(filtered_integrals[num_active - 1] == 1000.);

  CHECK(filtered_errors[0] == 10.);
  CHECK(filtered_errors[1] == 10.);
  CHECK(filtered_errors[num_active - 2] == 10.);
  CHECK(filtered_errors[num_active - 1] == 10.);

  CHECK(new_subdiv_dim[0] == 0);
  CHECK(new_subdiv_dim[1] == 0);
  CHECK(new_subdiv_dim[num_active - 2] == 0);
  CHECK(new_subdiv_dim[num_active - 1] == 0);

  CHECK(!regions_same(
    LeftCoord.data(), original_LeftCoord.data(), 0, 0, ndim, num_active, n));
  CHECK(regions_same(
    LeftCoord.data(), original_LeftCoord.data(), 0, 1, ndim, num_active, n));
  CHECK(!regions_same(LeftCoord.data(),
                      original_LeftCoord.data(),
                      num_active - 1,
                      n - 1,
                      ndim,
                      num_active,
                      n));
}
