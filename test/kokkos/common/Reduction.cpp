#include "catch2/catch.hpp"
#include "kokkos/pagani/quad/GPUquad/Sample.cuh"
#include "common/kokkos/cudaMemoryUtil.h"
#include "common/kokkos/Volume.cuh"
#include "common/kokkos/util.cuh"
#include "common/kokkos/custom_functions.cuh"
#include "common/kokkos/thrust_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace quad;

class PTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    double res = 15.37;
    return res;
  }
};

class NTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    double res = -15.37;
    return res;
  }
};

class ZTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    return 0.;
  }
};

TEST_CASE("Custom Reduction with Atomics")
{

  auto init_vector_and_compute_sum =
    [=](std::vector<double>& arr, double& val, size_t size) {
      arr.resize(size);
      std::iota(arr.begin(), arr.end(), val);
      double true_val = 0.;
      for (auto& v : arr) {
        true_val += v;
      }
      arr.clear();
      return true_val;
    };

  auto test_arrs_with_starting_val = [=](double val) {
    std::vector<double> arr;
    for (int i = 1; i <= 25; ++i) { // probably need to start with size 32 or
                                    // 64, I doubt it will work for 2, 4, 8, 16
      const size_t size = pow(2, i);
      const double true_val = init_vector_and_compute_sum(arr, val, size);
      ViewVectorDouble gpu_copy("temp", size);
      auto host_copy = Kokkos::create_mirror_view(gpu_copy);
      std::memcpy(host_copy.data(), arr.data(), sizeof(double) * size);
      Kokkos::deep_copy(gpu_copy, host_copy);
      const double res = custom_reduce(gpu_copy, size);
      CHECK(true_val == Approx(res));
    }
  };

  SECTION("Power 2 array sizes")
  {
    test_arrs_with_starting_val(-1000.);
    test_arrs_with_starting_val(1000.);
    test_arrs_with_starting_val(1000.5);
    test_arrs_with_starting_val(1.);
  }

  SECTION("Non-power 2 array sizes")
  {
    double vec_first_val = 1.;
    std::vector<double> arr;
    for (int i = 1; i < 1000; ++i) { // probably need to start with size 32 or
                                     // 64, I doubt it will work for 2, 4, 8, 16
      size_t size = i + 10;
      const double true_val =
        init_vector_and_compute_sum(arr, vec_first_val, size);
      ViewVectorDouble gpu_copy = cuda_malloc<double>(size);
      auto host_copy = Kokkos::create_mirror_view(gpu_copy);
      std::memcpy(host_copy.data(), arr.data(), sizeof(double) * size);
      Kokkos::deep_copy(gpu_copy, host_copy);
      const double res = custom_reduce(gpu_copy, size);
      CHECK(true_val == Approx(res));
    }
  }
}

TEST_CASE("Custom Reduction with Atomics - Common Inteface with Thrust")
{
  constexpr bool use_custom = true;
  auto init_vector_and_compute_sum =
    [=](std::vector<double>& arr, double& val, size_t size) {
      arr.resize(size);
      std::iota(arr.begin(), arr.end(), val);
      double true_val = 0.;
      for (auto& v : arr) {
        true_val += v;
      }
      arr.clear();
      return true_val;
    };

  auto test_arrs_with_starting_val = [=](double val) {
    std::vector<double> arr;
    for (int i = 1; i <= 25; ++i) { // probably need to start with size 32 or
                                    // 64, I doubt it will work for 2, 4, 8, 16
      const size_t size = pow(2, i);
      const double true_val = init_vector_and_compute_sum(arr, val, size);
      ViewVectorDouble gpu_copy("temp", size);
      auto host_copy = Kokkos::create_mirror_view(gpu_copy);
      std::memcpy(host_copy.data(), arr.data(), sizeof(double) * size);
      Kokkos::deep_copy(gpu_copy, host_copy);
      const double res = reduction<double, use_custom>(gpu_copy, size);
      CHECK(true_val == Approx(res));
    }
  };

  SECTION("Power 2 array sizes")
  {
    test_arrs_with_starting_val(-1000.);
    test_arrs_with_starting_val(1000.);
    test_arrs_with_starting_val(1000.5);
    test_arrs_with_starting_val(1.);
  }

  SECTION("Non-power 2 array sizes")
  {
    double vec_first_val = 1.;
    std::vector<double> arr;
    for (int i = 1; i < 1000; ++i) { // probably need to start with size 32 or
                                     // 64, I doubt it will work for 2, 4, 8, 16
      size_t size = i + 10;
      const double true_val =
        init_vector_and_compute_sum(arr, vec_first_val, size);
      ViewVectorDouble gpu_copy("temp", size);
      auto host_copy = Kokkos::create_mirror_view(gpu_copy);
      std::memcpy(host_copy.data(), arr.data(), sizeof(double) * size);
      Kokkos::deep_copy(gpu_copy, host_copy);
      const double res = reduction<double, use_custom>(gpu_copy, size);
      CHECK(true_val == Approx(res));
    }
  }
}
