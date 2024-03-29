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

using namespace quad;

class PTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    double res = 15.37;
    return res;
  }
};

class NTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    double res = -15.37;
    return res;
  }
};

class ZTest {
public:
  __device__ __host__ double
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
      double* gpu_copy = cuda_malloc<double>(size);
      cuda_memcpy_to_device<double>(gpu_copy, arr.data(), size);
      const double res = custom_reduce_atomics(gpu_copy, size);
      cudaFree(gpu_copy);
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
      double* gpu_copy = cuda_malloc<double>(size);
      cuda_memcpy_to_device<double>(gpu_copy, arr.data(), size);
      const double res = custom_reduce_atomics(gpu_copy, size);
      cudaFree(gpu_copy);
      CHECK(true_val == Approx(res));
    }
  }
}

TEST_CASE("Custom Reduction with Atomics - Common Inteface with Thrust")
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
      double* gpu_copy = cuda_malloc<double>(size);
      cuda_memcpy_to_device<double>(gpu_copy, arr.data(), size);
      const double res = reduction<double, false>(gpu_copy, size);
      cudaFree(gpu_copy);
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
      double* gpu_copy = cuda_malloc<double>(size);
      cuda_memcpy_to_device<double>(gpu_copy, arr.data(), size);
      const double res = reduction<double, false>(gpu_copy, size);
      cudaFree(gpu_copy);
      CHECK(true_val == Approx(res));
    }
  }
}
