#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include <iostream>
#include "oneAPI/pagani/quad/GPUquad/PaganiUtils.dp.hpp"
#include "common/oneAPI/cudaMemoryUtil.h"
#include "common/oneAPI/thrust_utils.dp.hpp"
#include "common/oneAPI/custom_functions.dp.hpp"

TEST_CASE("Custom Reduction with Atomics")
{
  auto q_ct1 = sycl::queue(sycl::gpu_selector());
  ;
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
      double* gpu_copy = quad::cuda_malloc<double>(size);
      quad::cuda_memcpy_to_device<double>(gpu_copy, arr.data(), size);
      const double res = custom_reduce_atomics(gpu_copy, size);
      sycl::free(gpu_copy, q_ct1);
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
      double* gpu_copy = quad::cuda_malloc<double>(size);
      quad::cuda_memcpy_to_device<double>(gpu_copy, arr.data(), size);
      const double res = custom_reduce_atomics(gpu_copy, size);
      sycl::free(gpu_copy, q_ct1);
      CHECK(true_val == Approx(res));
    }
  }
}

TEST_CASE("Thrust Reduction")
{
  auto q_ct1 = sycl::queue(sycl::gpu_selector());
  ;
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
      double* gpu_copy = quad::cuda_malloc<double>(size);
      quad::cuda_memcpy_to_device<double>(gpu_copy, arr.data(), size);
      const double res = reduction<double>(gpu_copy, size);
      sycl::free(gpu_copy, q_ct1);
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
      double* gpu_copy = quad::cuda_malloc<double>(size);
      quad::cuda_memcpy_to_device<double>(gpu_copy, arr.data(), size);
      const double res = reduction<double>(gpu_copy, size);
      sycl::free(gpu_copy, q_ct1);
      CHECK(true_val == Approx(res));
    }
  }
}