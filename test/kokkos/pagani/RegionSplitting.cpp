#include "catch2/catch.hpp"
#include "kokkos/pagani/quad/quad.h"
#include "common/kokkos/cudaMemoryUtil.h"
#include "common/kokkos/Volume.cuh"
// #include <chrono>
// #include <cmath>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <numeric>

#include "kokkos/pagani/quad/GPUquad/Sub_regions.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_region_splitter.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"

// #include "common/integration_result.hh"

/*#include "kokkos/pagani/quad/GPUquad/Sub_regions.cuh"
#include "catch2/catch.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>*/

template <size_t ndim>
bool
is_free_of_duplicates(Sub_regions<double, ndim>& regions)
{
  for (size_t regionID = 0; regionID < regions.size; regionID++) {
    quad::Volume<double, ndim> region = regions.extract_region(regionID);
    for (size_t reg = 0; reg < regions.size; reg++) {
      quad::Volume<double, ndim> region_i = regions.extract_region(reg);
      if (reg != regionID && region == region_i) {
        return false;
      }
    }
  }
  return true;
}

TEST_CASE("Split all regions at dim 1")
{
  constexpr int ndim = 2;
  Sub_regions<double, ndim> regions(2);
  const size_t n = regions.size;

  Sub_region_splitter<double, ndim> splitter(n);
  Region_characteristics<ndim> classifications(n);

  auto sub_div_dim =
    Kokkos::create_mirror_view(classifications.sub_dividing_dim);
  auto orig_leftcoord = Kokkos::create_mirror_view(regions.dLeftCoord);
  auto orig_length = Kokkos::create_mirror_view(regions.dLength);

  Kokkos::deep_copy(orig_leftcoord, regions.dLeftCoord);
  Kokkos::deep_copy(orig_length, regions.dLength);

  for (size_t i = 0; i < n; ++i) {
    sub_div_dim[i] = 1;
  }

  Kokkos::deep_copy(classifications.sub_dividing_dim, sub_div_dim);
  splitter.split(regions, classifications);

  auto new_leftcoord = Kokkos::create_mirror_view(regions.dLeftCoord);
  auto new_length = Kokkos::create_mirror_view(regions.dLength);

  Kokkos::deep_copy(new_leftcoord, regions.dLeftCoord);
  Kokkos::deep_copy(new_length, regions.dLength);

  SECTION("Dimension zero is intact")
  {
    for (size_t i = 0; i < n; ++i) {
      const size_t dim = 0;
      const size_t par_index = i + dim * n;
      const size_t left = i + dim * n * 2;
      const size_t right = n + i + dim * n * 2;

      CHECK(new_leftcoord[left] == Approx(orig_leftcoord[par_index]));
      CHECK(new_leftcoord[right] == Approx(orig_leftcoord[par_index]));

      CHECK(new_length[left] == Approx(orig_length[par_index]));
      CHECK(new_length[right] == Approx(orig_length[par_index]));
    }
  }

  SECTION("Dimension one is changed")
  {
    for (size_t i = 0; i < n; ++i) {
      const size_t dim = 1;
      const size_t par_index = i + dim * n;
      const size_t left = i + dim * n * 2;
      const size_t right = n + i + dim * n * 2;

      CHECK(new_leftcoord[left] == Approx(orig_leftcoord[par_index]));
      CHECK(new_leftcoord[right] ==
            Approx(orig_leftcoord[par_index] + orig_length[par_index] / 2));

      CHECK(new_length[left] == Approx(orig_length[par_index] / 2));
      CHECK(new_length[right] == Approx(orig_length[par_index] / 2));
    }
  }
}

TEST_CASE("Split first region at dim 0 the rest at dim 1")
{
  constexpr int ndim = 2;
  Sub_regions<double, ndim> regions(5);
  const size_t n = regions.size;

  Sub_region_splitter<double, ndim> splitter(n);
  Region_characteristics<ndim> classifications(n);

  auto sub_div_dim =
    Kokkos::create_mirror_view(classifications.sub_dividing_dim);
  auto orig_leftcoord = Kokkos::create_mirror_view(regions.dLeftCoord);
  auto orig_length = Kokkos::create_mirror_view(regions.dLength);

  Kokkos::deep_copy(orig_leftcoord, regions.dLeftCoord);
  Kokkos::deep_copy(orig_length, regions.dLength);

  sub_div_dim[0] = 0;
  for (size_t i = 1; i < n; ++i) {
    sub_div_dim[i] = 1;
  }

  Kokkos::deep_copy(classifications.sub_dividing_dim, sub_div_dim);
  splitter.split(regions, classifications);

  auto new_leftcoord = Kokkos::create_mirror_view(regions.dLeftCoord);
  auto new_length = Kokkos::create_mirror_view(regions.dLength);

  Kokkos::deep_copy(new_leftcoord, regions.dLeftCoord);
  Kokkos::deep_copy(new_length, regions.dLength);

  SECTION("Dimension zero is changed only for the first region")
  {
    // check regions split at dim 1
    for (size_t reg = 1; reg < n; ++reg) {

      const size_t dim = 0;
      const size_t par_index = reg + dim * n;
      const size_t left = reg + dim * n * 2;
      const size_t right = n + reg + dim * n * 2;

      if (reg == 0) {
        CHECK(new_leftcoord[left] == Approx(orig_leftcoord[par_index]));
        CHECK(new_leftcoord[right] ==
              Approx(orig_leftcoord[par_index] + orig_length[par_index] / 2));

        CHECK(new_length[left] == Approx(orig_length[par_index] / 2));
        CHECK(new_length[right] == Approx(orig_length[par_index] / 2));
      }

      if (reg != 0) {
        CHECK(new_leftcoord[left] == Approx(orig_leftcoord[par_index]));
        CHECK(new_leftcoord[right] == Approx(orig_leftcoord[par_index]));

        CHECK(new_length[left] == Approx(orig_length[par_index]));
        CHECK(new_length[right] == Approx(orig_length[par_index]));
      }
    }
  }

  SECTION("Dimension one is changed for all but the first region")
  {
    for (size_t reg = 0; reg < n; ++reg) {
      const size_t dim = 1;
      const size_t par_index = reg + dim * n;
      const size_t left = reg + dim * n * 2;
      const size_t right = n + reg + dim * n * 2;

      if (reg == 0) {
        CHECK(new_leftcoord[left] == Approx(orig_leftcoord[par_index]));
        CHECK(new_leftcoord[right] == Approx(orig_leftcoord[par_index]));

        CHECK(new_length[left] == Approx(orig_length[par_index]));
        CHECK(new_length[right] == Approx(orig_length[par_index]));
      }

      if (reg != 0) {
        CHECK(new_leftcoord[left] == Approx(orig_leftcoord[par_index]));
        CHECK(new_leftcoord[right] ==
              Approx(orig_leftcoord[par_index] + orig_length[par_index] / 2));

        CHECK(new_length[left] == Approx(orig_length[par_index] / 2));
        CHECK(new_length[right] == Approx(orig_length[par_index] / 2));
      }
    }
  }
}
