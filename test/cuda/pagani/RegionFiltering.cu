#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/demos/function.cuh"
#include "cuda/pagani/quad/GPUquad/Sample.cuh"
#include "cuda/pagani/quad/quad.h"
#include "common/cuda/cudaMemoryUtil.h"
#include "common/cuda/Volume.cuh"
#include "common/cuda/cudaUtil.h"
#include "common/cuda/custom_functions.cuh"
#include "common/cuda/thrust_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "cuda/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"
#include "common/cuda/integrands.cuh"
#include "common/integration_result.hh"

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

  double* host_active_regions = quad::host_alloc<double>(num_regions);

  SECTION("All active regions")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 1;

    quad::cuda_memcpy_to_device<double>(
      regions.active_regions, host_active_regions, num_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == num_regions);
  }

  SECTION("All but last region is active")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 0;
    host_active_regions[regions.size - 1] = 1;

    quad::cuda_memcpy_to_device<double>(
      regions.active_regions, host_active_regions, num_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 1);
  }

  SECTION("All but the first region is active")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 0;
    host_active_regions[0] = 1;

    quad::cuda_memcpy_to_device<double>(
      regions.active_regions, host_active_regions, num_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 1);
  }

  SECTION("Zero active regions")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 0;

    quad::cuda_memcpy_to_device<double>(
      regions.active_regions, host_active_regions, num_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 0);
  }

  SECTION("Only first region is inactive")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 1;
    host_active_regions[0] = 0;

    quad::cuda_memcpy_to_device<double>(
      regions.active_regions, host_active_regions, num_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 99);
  }

  SECTION("Only last region is inactive")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 1;
    host_active_regions[regions.size - 1] = 0;

    quad::cuda_memcpy_to_device<double>(
      regions.active_regions, host_active_regions, num_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 99);
  }

  delete[] host_active_regions;
};

TEST_CASE("get_num_active_regions: one-region-list")
{
  constexpr int ndim = 2;
  constexpr size_t num_regions = 1;
  Region_characteristics<ndim> regions(num_regions);
  Sub_regions_filter<double, ndim> filter(num_regions);

  double* host_active_regions = quad::host_alloc<double>(num_regions);

  SECTION("one active region")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 1;
    host_active_regions[0] = 1;

    quad::cuda_memcpy_to_device<double>(
      regions.active_regions, host_active_regions, num_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 1);
  }

  SECTION("zero active regions")
  {
    for (size_t i = 0; i < num_regions; ++i)
      host_active_regions[i] = 0;

    quad::cuda_memcpy_to_device<double>(
      regions.active_regions, host_active_regions, num_regions);
    size_t num_active =
      filter.get_num_active_regions(regions.active_regions, regions.size);
    CHECK(num_active == 0);
  }

  delete[] host_active_regions;
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

  double* integrals_mirror = quad::host_alloc<double>(n);
  double* error_mirror = quad::host_alloc<double>(n);
  double* parent_integrals_mirror = quad::host_alloc<double>(n);
  double* parent_error_mirror = quad::host_alloc<double>(n);
  double* original_LeftCoord = quad::host_alloc<double>(n * ndim);
  double* original_Length = quad::host_alloc<double>(n * ndim);
  int* sub_dividing_dim = quad::host_alloc<int>(n);
  double* host_active_regions = quad::host_alloc<double>(n);

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

  quad::cuda_memcpy_to_host<double>(
    original_LeftCoord, coordinates.dLeftCoord, n * ndim);
  quad::cuda_memcpy_to_host<double>(
    original_Length, coordinates.dLength, n * ndim);

  quad::cuda_memcpy_to_device<double>(
    estimates.integral_estimates, integrals_mirror, n);
  quad::cuda_memcpy_to_device<double>(
    estimates.error_estimates, error_mirror, n);
  quad::cuda_memcpy_to_device<double>(
    parents.integral_estimates, parent_integrals_mirror, n / 2);
  quad::cuda_memcpy_to_device<double>(
    parents.error_estimates, parent_error_mirror, n / 2);
  quad::cuda_memcpy_to_device<int>(
    classifications.sub_dividing_dim, sub_dividing_dim, n);
  quad::cuda_memcpy_to_device<double>(
    classifications.active_regions, host_active_regions, n);

  size_t num_active =
    filter.filter(coordinates, classifications, estimates, parents);
  CHECK(num_active == n);

  double* LeftCoord = quad::host_alloc<double>(num_active * ndim);
  double* Length = quad::host_alloc<double>(num_active * ndim);
  double* filtered_integrals = quad::host_alloc<double>(num_active);
  double* filtered_errors = quad::host_alloc<double>(num_active);
  int* new_subdiv_dim = quad::host_alloc<int>(num_active);

  // after filter, the pre-filtering integral and error-estimates become the
  // parents
  quad::cuda_memcpy_to_host<double>(
    filtered_integrals, parents.integral_estimates, num_active);
  quad::cuda_memcpy_to_host<double>(
    filtered_errors, parents.error_estimates, num_active);
  quad::cuda_memcpy_to_host<double>(
    LeftCoord, coordinates.dLeftCoord, num_active * ndim);
  quad::cuda_memcpy_to_host<double>(
    Length, coordinates.dLength, num_active * ndim);
  quad::cuda_memcpy_to_host<int>(
    new_subdiv_dim, classifications.sub_dividing_dim, num_active);

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
    CHECK(
      regions_same(LeftCoord, original_LeftCoord, i, i, ndim, num_active, n));

  delete[] Length;
  delete[] LeftCoord;
  delete[] integrals_mirror;
  delete[] error_mirror;
  delete[] host_active_regions;
  delete[] parent_integrals_mirror;
  delete[] parent_error_mirror;
  delete[] filtered_integrals;
  delete[] filtered_errors;
  delete[] original_LeftCoord;
  delete[] original_Length;
  delete[] sub_dividing_dim;
  delete[] new_subdiv_dim;
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

  double* integrals_mirror = quad::host_alloc<double>(n);
  double* error_mirror = quad::host_alloc<double>(n);
  double* parent_integrals_mirror = quad::host_alloc<double>(n);
  double* parent_error_mirror = quad::host_alloc<double>(n);
  double* original_LeftCoord = quad::host_alloc<double>(n * ndim);
  double* original_Length = quad::host_alloc<double>(n * ndim);
  int* sub_dividing_dim = quad::host_alloc<int>(n);
  double* host_active_regions = quad::host_alloc<double>(n);

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

  quad::cuda_memcpy_to_host<double>(
    original_LeftCoord, coordinates.dLeftCoord, n * ndim);
  quad::cuda_memcpy_to_host<double>(
    original_Length, coordinates.dLength, n * ndim);

  quad::cuda_memcpy_to_device<double>(
    estimates.integral_estimates, integrals_mirror, n);
  quad::cuda_memcpy_to_device<double>(
    estimates.error_estimates, error_mirror, n);
  quad::cuda_memcpy_to_device<double>(
    parents.integral_estimates, parent_integrals_mirror, n / 2);
  quad::cuda_memcpy_to_device<double>(
    parents.error_estimates, parent_error_mirror, n / 2);
  quad::cuda_memcpy_to_device<int>(
    classifications.sub_dividing_dim, sub_dividing_dim, n);
  quad::cuda_memcpy_to_device<double>(
    classifications.active_regions, host_active_regions, n);

  size_t num_active =
    filter.filter(coordinates, classifications, estimates, parents);
  CHECK(num_active == n - 1);

  // after filter, the pre-filtering integral and error-estimates become the
  // parents
  double* LeftCoord = quad::host_alloc<double>(num_active * ndim);
  double* Length = quad::host_alloc<double>(num_active * ndim);
  double* filtered_integrals = quad::host_alloc<double>(num_active);
  double* filtered_errors = quad::host_alloc<double>(num_active);
  int* new_subdiv_dim = quad::host_alloc<int>(num_active);

  // after filter, the pre-filtering integral and error-estimates become the
  // parents
  quad::cuda_memcpy_to_host<double>(
    filtered_integrals, parents.integral_estimates, num_active);
  quad::cuda_memcpy_to_host<double>(
    filtered_errors, parents.error_estimates, num_active);
  quad::cuda_memcpy_to_host<double>(
    LeftCoord, coordinates.dLeftCoord, num_active * ndim);
  quad::cuda_memcpy_to_host<double>(
    Length, coordinates.dLength, num_active * ndim);
  quad::cuda_memcpy_to_host<int>(
    new_subdiv_dim, classifications.sub_dividing_dim, num_active);

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
  CHECK(regions_same(LeftCoord, original_LeftCoord, 0, 0, ndim, num_active, n));
  CHECK(regions_same(LeftCoord, original_LeftCoord, 1, 1, ndim, num_active, n));
  CHECK(
    regions_same(LeftCoord, original_LeftCoord, 14, 14, ndim, num_active, n));
  CHECK(
    !regions_same(LeftCoord, original_LeftCoord, 15, 15, ndim, num_active, n));
  CHECK(
    regions_same(LeftCoord, original_LeftCoord, 15, 16, ndim, num_active, n));
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

  double* integrals_mirror = quad::host_alloc<double>(n);
  double* error_mirror = quad::host_alloc<double>(n);
  double* parent_integrals_mirror = quad::host_alloc<double>(n);
  double* parent_error_mirror = quad::host_alloc<double>(n);
  double* original_LeftCoord = quad::host_alloc<double>(n * ndim);
  double* original_Length = quad::host_alloc<double>(n * ndim);
  int* sub_dividing_dim = quad::host_alloc<int>(n);
  double* host_active_regions = quad::host_alloc<double>(n);

  for (int i = 0; i < n; ++i) {
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

  quad::cuda_memcpy_to_host<double>(
    original_LeftCoord, coordinates.dLeftCoord, n * ndim);
  quad::cuda_memcpy_to_host<double>(
    original_Length, coordinates.dLength, n * ndim);

  quad::cuda_memcpy_to_device<double>(
    estimates.integral_estimates, integrals_mirror, n);
  quad::cuda_memcpy_to_device<double>(
    estimates.error_estimates, error_mirror, n);
  quad::cuda_memcpy_to_device<double>(
    parents.integral_estimates, parent_integrals_mirror, n / 2);
  quad::cuda_memcpy_to_device<double>(
    parents.error_estimates, parent_error_mirror, n / 2);
  quad::cuda_memcpy_to_device<int>(
    classifications.sub_dividing_dim, sub_dividing_dim, n);
  quad::cuda_memcpy_to_device<double>(
    classifications.active_regions, host_active_regions, n);

  size_t num_active =
    filter.filter(coordinates, classifications, estimates, parents);
  CHECK(num_active == n - 2);

  // after filter, the pre-filtering integral and error-estimates become the
  // parents
  double* LeftCoord = quad::host_alloc<double>(num_active * ndim);
  double* Length = quad::host_alloc<double>(num_active * ndim);
  double* filtered_integrals = quad::host_alloc<double>(num_active);
  double* filtered_errors = quad::host_alloc<double>(num_active);
  int* new_subdiv_dim = quad::host_alloc<int>(num_active);

  // after filter, the pre-filtering integral and error-estimates become the
  // parents
  quad::cuda_memcpy_to_host<double>(
    filtered_integrals, parents.integral_estimates, num_active);
  quad::cuda_memcpy_to_host<double>(
    filtered_errors, parents.error_estimates, num_active);
  quad::cuda_memcpy_to_host<double>(
    LeftCoord, coordinates.dLeftCoord, num_active * ndim);
  quad::cuda_memcpy_to_host<double>(
    Length, coordinates.dLength, num_active * ndim);
  quad::cuda_memcpy_to_host<int>(
    new_subdiv_dim, classifications.sub_dividing_dim, num_active);

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

  CHECK(
    !regions_same(LeftCoord, original_LeftCoord, 0, 0, ndim, num_active, n));
  CHECK(regions_same(LeftCoord, original_LeftCoord, 0, 1, ndim, num_active, n));
  CHECK(!regions_same(
    LeftCoord, original_LeftCoord, num_active - 1, n - 1, ndim, num_active, n));
}
