#include "kokkos/pagani/quad/Cuhre.cuh"
#include "catch.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

TEST_CASE("Unit-Volume 2D")
{
  constexpr int NDIM = 2;
  Kernel<double, NDIM> kernel(0);

  ViewVectorDouble dRegions("dRegions", NDIM);
  ViewVectorDouble dRegionsLength("dRegionsLength", NDIM);

  kernel.GenerateInitialRegions(dRegions, dRegionsLength);

  SECTION("Total Volume of GPU regions is 1.0")
  {
    double gpu_volume = 0.;
    Kokkos::parallel_reduce(
      "Sub-volume Reduction",
      kernel.numRegions,
      KOKKOS_LAMBDA(const int index, double& valueToUpdate) {
        double sub_region_volume = 1.;
        for (int dim = 0; dim < NDIM; ++dim) {
          sub_region_volume *= dRegionsLength(dim * kernel.numRegions + index);
        }
        valueToUpdate += sub_region_volume;
      },
      gpu_volume);

    CHECK(gpu_volume == 1.);
  }

  ViewVectorDouble::HostMirror Regions = Kokkos::create_mirror_view(dRegions);
  ViewVectorDouble::HostMirror RegionsLength =
    Kokkos::create_mirror_view(dRegionsLength);
  Kokkos::deep_copy(Regions, dRegions);
  Kokkos::deep_copy(RegionsLength, dRegionsLength);

  SECTION("Region Data Properly Transferred to CPU")
  {
    double cpu_volume = 0.;
    for (size_t index = 0; index < kernel.numRegions; ++index) {
      double sub_region_volume = 1.;
      for (int dim = 0; dim < NDIM; ++dim) {
        sub_region_volume *= RegionsLength(dim * kernel.numRegions + index);
      }
      cpu_volume += sub_region_volume;
    }
    CHECK(cpu_volume == 1.);
  }
};

TEST_CASE("Unit-Volume 5D")
{

  constexpr int NDIM = 5;
  // int NSETS = 0; //variable unused for the purposes of this test
  Kernel<double, NDIM> kernel(0);

  ViewVectorDouble dRegions("dRegions", NDIM);
  ViewVectorDouble dRegionsLength("dRegionsLength", NDIM);

  kernel.GenerateInitialRegions(dRegions, dRegionsLength);

  SECTION("Total Volume of GPU regions is 1.0")
  {
    double gpu_volume = 0.;
    Kokkos::parallel_reduce(
      "Sub-volume Reduction",
      kernel.numRegions,
      KOKKOS_LAMBDA(const int index, double& valueToUpdate) {
        double sub_region_volume = 1.;
        for (int dim = 0; dim < NDIM; dim++) {
          sub_region_volume *= dRegionsLength(dim * kernel.numRegions + index);
        }
        valueToUpdate += sub_region_volume;
      },
      gpu_volume);

    CHECK(gpu_volume == 1.);
  }

  ViewVectorDouble::HostMirror Regions = Kokkos::create_mirror_view(dRegions);
  ViewVectorDouble::HostMirror RegionsLength =
    Kokkos::create_mirror_view(dRegionsLength);
  Kokkos::deep_copy(Regions, dRegions);
  Kokkos::deep_copy(RegionsLength, dRegionsLength);

  SECTION("Region Data Properly Transferred to CPU")
  {
    double cpu_volume = 0.;
    for (size_t index = 0; index < kernel.numRegions; index++) {
      double sub_region_volume = 1.;
      for (int dim = 0; dim < NDIM; dim++) {
        sub_region_volume *= RegionsLength(dim * kernel.numRegions + index);
      }
      cpu_volume += sub_region_volume;
    }
    CHECK(cpu_volume == 1.);
  }
};

TEST_CASE("Unit-Volume 10D")
{
  constexpr int NDIM = 10;
  // int NSETS = 0; //variable unused for the purposes of this test
  Kernel<double, NDIM> kernel(0);

  ViewVectorDouble dRegions("dRegions", NDIM);
  ViewVectorDouble dRegionsLength("dRegionsLength", NDIM);

  kernel.GenerateInitialRegions(dRegions, dRegionsLength);

  SECTION("Total Volume of GPU regions is 1.0")
  {
    double gpu_volume = 0.;
    Kokkos::parallel_reduce(
      "Sub-volume Reduction",
      kernel.numRegions,
      KOKKOS_LAMBDA(const int index, double& valueToUpdate) {
        double sub_region_volume = 1.;
        for (int dim = 0; dim < NDIM; dim++) {
          sub_region_volume *= dRegionsLength(dim * kernel.numRegions + index);
        }
        valueToUpdate += sub_region_volume;
      },
      gpu_volume);

    printf("Total Volume:%f\n", gpu_volume);
    CHECK(gpu_volume == 1.);
  }

  ViewVectorDouble::HostMirror Regions = Kokkos::create_mirror_view(dRegions);
  ViewVectorDouble::HostMirror RegionsLength =
    Kokkos::create_mirror_view(dRegionsLength);
  Kokkos::deep_copy(Regions, dRegions);
  Kokkos::deep_copy(RegionsLength, dRegionsLength);

  SECTION("Region Data Properly Transferred to CPU")
  {
    double cpu_volume = 0.;
    for (size_t index = 0; index < kernel.numRegions; index++) {
      double sub_region_volume = 1.;
      for (int dim = 0; dim < NDIM; dim++) {
        sub_region_volume *= RegionsLength(dim * kernel.numRegions + index);
      }
      cpu_volume += sub_region_volume;
    }
    CHECK(cpu_volume == 1.);
  }
};
