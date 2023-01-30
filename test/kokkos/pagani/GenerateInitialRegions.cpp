#include "kokkos/pagani/quad/Cuhre.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_regions.cuh"
#include "catch2/catch.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

TEST_CASE("New Unit-Volume 2D")
{
  constexpr int ndim = 2;
  Kernel<double, ndim> kernel(0);

  Sub_regions<double, ndim> regions(2);

  SECTION("Total Volume of GPU regions is 1.0")
  {
    double gpu_volume = regions.compute_total_volume();
    CHECK(gpu_volume == 1.);
  }

  ViewVectorDouble::HostMirror Regions =
    Kokkos::create_mirror_view(regions.dLeftCoord);
  ViewVectorDouble::HostMirror RegionsLength =
    Kokkos::create_mirror_view(regions.dLength);
  Kokkos::deep_copy(Regions, regions.dLeftCoord);
  Kokkos::deep_copy(RegionsLength, regions.dLength);

  SECTION("Region Data Properly Transferred to CPU")
  {
    double cpu_volume = 0.;
    for (size_t index = 0; index < regions.size; ++index) {
      double sub_region_volume = 1.;
      for (int dim = 0; dim < ndim; ++dim) {
        sub_region_volume *= RegionsLength(dim * regions.size + index);

        CHECK(Regions[dim * regions.size + index] >= 0.);
        CHECK(RegionsLength[dim * regions.size + index] < 1.);
        CHECK(RegionsLength[dim * regions.size + index] > 0.);
      }

      CHECK(sub_region_volume < 1.);
      CHECK(sub_region_volume > 0.);
      CHECK(sub_region_volume ==
            Approx(1. / static_cast<double>(regions.size)));

      cpu_volume += sub_region_volume;
    }
    CHECK(cpu_volume == 1.);
  }
};

TEST_CASE("New Unit-Volume 5D")
{
  constexpr int ndim = 5;
  Kernel<double, ndim> kernel(0);

  Sub_regions<double, ndim> regions(3);

  SECTION("Total Volume of GPU regions is 1.0")
  {
    double gpu_volume = regions.compute_total_volume();
    CHECK(gpu_volume == Approx(1.).epsilon(1.e-9));
  }

  ViewVectorDouble::HostMirror Regions =
    Kokkos::create_mirror_view(regions.dLeftCoord);
  ViewVectorDouble::HostMirror RegionsLength =
    Kokkos::create_mirror_view(regions.dLength);
  Kokkos::deep_copy(Regions, regions.dLeftCoord);
  Kokkos::deep_copy(RegionsLength, regions.dLength);

  SECTION("Region Data Properly Transferred to CPU")
  {
    double cpu_volume = 0.;
    for (size_t index = 0; index < regions.size; ++index) {
      double sub_region_volume = 1.;
      for (int dim = 0; dim < ndim; ++dim) {
        sub_region_volume *= RegionsLength(dim * regions.size + index);
        CHECK(Regions[dim * regions.size + index] >= 0.);
        CHECK(RegionsLength[dim * regions.size + index] < 1.);
        CHECK(RegionsLength[dim * regions.size + index] > 0.);
      }

      CHECK(sub_region_volume < 1.);
      CHECK(sub_region_volume > 0.);
      CHECK(sub_region_volume ==
            Approx(1. / static_cast<double>(regions.size)));
      cpu_volume += sub_region_volume;
    }
    CHECK(cpu_volume == Approx(1.).epsilon(1.e-9));
  }
};

TEST_CASE("New Unit-Volume 8D")
{
  constexpr int ndim = 8;
  Kernel<double, ndim> kernel(0);

  Sub_regions<double, ndim> regions(4);

  SECTION("Total Volume of GPU regions is 1.0")
  {
    double gpu_volume = regions.compute_total_volume();
    CHECK(gpu_volume == Approx(1.).epsilon(1.e-9));
  }

  ViewVectorDouble::HostMirror Regions =
    Kokkos::create_mirror_view(regions.dLeftCoord);
  ViewVectorDouble::HostMirror RegionsLength =
    Kokkos::create_mirror_view(regions.dLength);
  Kokkos::deep_copy(Regions, regions.dLeftCoord);
  Kokkos::deep_copy(RegionsLength, regions.dLength);

  SECTION("Region Data Properly Transferred to CPU")
  {
    double cpu_volume = 0.;
    for (size_t index = 0; index < regions.size; ++index) {
      double sub_region_volume = 1.;
      for (int dim = 0; dim < ndim; ++dim) {
        sub_region_volume *= RegionsLength(dim * regions.size + index);
        CHECK(Regions[dim * regions.size + index] >= 0.);
        CHECK(RegionsLength[dim * regions.size + index] < 1.);
        CHECK(RegionsLength[dim * regions.size + index] > 0.);
      }
      CHECK(sub_region_volume < 1.);
      CHECK(sub_region_volume > 0.);
      CHECK(sub_region_volume ==
            Approx(1. / static_cast<double>(regions.size)));
      cpu_volume += sub_region_volume;
    }
    CHECK(cpu_volume == Approx(1.).epsilon(1.e-9));
  }
};
