#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/demos/function.cuh"
#include "cuda/pagani/quad/quad.h"
#include "common/cuda/Volume.cuh"
#include "common/cuda/cudaUtil.h"
#include "common/cuda/cudaMemoryUtil.h"
#include "common/cuda/custom_functions.cuh"
#include <utility>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <array>

template <typename T, size_t size>
T*
make_gpu_arr(std::array<T, size> arr)
{
  double* d_arr = quad::cuda_malloc_managed<double>(arr.size());
  quad::cuda_memcpy_to_device<double>(d_arr, arr.data(), arr.size());
  return d_arr;
}

TEST_CASE("Half Block")
{
  constexpr size_t size = 512;
  std::array<double, size> arr;
  std::fill(arr.begin(), arr.end(), 3.9);

  using MinMax = std::pair<double, double>;

  SECTION("Testing min at different positions")
  {
    arr[0] = 1.;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);

    arr[size - 1] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);

    arr[size / 2] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);

    arr[size / 4] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
  }

  SECTION("Testing max at different positions")
  {
    arr[0] = 4.1;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[0] = 1.;

    arr[size - 1] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size - 1] = 1.;

    arr[size / 2] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size / 2] = 1.;

    arr[size / 4] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
  }
}

TEST_CASE("Full Block")
{
  constexpr size_t size = 1024;
  std::array<double, size> arr;
  std::fill(arr.begin(), arr.end(), 3.9);

  using MinMax = std::pair<double, double>;

  SECTION("Testing min at different positions")
  {
    arr[0] = 1.;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[0] = 3.9;

    arr[size - 1] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[size - 1] = 3.9;

    arr[size / 2] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[size / 2] = 3.9;

    arr[size / 4] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
  }

  SECTION("Testing max at different positions")
  {
    arr[0] = 4.1;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[0] = 3.9;

    arr[size - 1] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size - 1] = 3.9;

    arr[size / 2] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size / 2] = 3.9;

    arr[size / 4] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size / 4] = 3.9;
  }
}

TEST_CASE("Two Full Blocks")
{
  constexpr size_t size = 2048;
  std::array<double, size> arr;
  std::fill(arr.begin(), arr.end(), 3.9);

  using MinMax = std::pair<double, double>;

  SECTION("Testing min at different positions")
  {
    arr[0] = 1.;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[0] = 3.9;

    arr[size - 1] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[size - 1] = 3.9;

    arr[size / 2] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[size / 2] = 3.9;

    arr[size / 4] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
  }

  SECTION("Testing max at different positions")
  {
    arr[0] = 4.1;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[0] = 3.9;

    arr[size - 1] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size - 1] = 3.9;

    arr[size / 2] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size / 2] = 3.9;

    arr[size / 4] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size / 4] = 3.9;
  }
}

TEST_CASE("Misaligned Partial Block")
{
  constexpr size_t size = 1000;
  std::array<double, size> arr;
  std::fill(arr.begin(), arr.end(), 3.9);

  using MinMax = std::pair<double, double>;

  SECTION("Testing min at different positions")
  {
    arr[0] = 1.;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[0] = 3.9;

    arr[size - 1] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[size - 1] = 3.9;

    arr[size / 2] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[size / 2] = 3.9;

    arr[size / 4] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
  }

  SECTION("Testing max at different positions")
  {
    arr[0] = 4.1;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[0] = 3.9;

    arr[size - 1] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size - 1] = 3.9;

    arr[size / 2] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size / 2] = 3.9;

    arr[size / 4] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
  }
}

TEST_CASE("Misaligned Partial Block with multiple block launches")
{
  constexpr size_t size = 2052;
  std::array<double, size> arr;
  std::fill(arr.begin(), arr.end(), 3.9);

  using MinMax = std::pair<double, double>;

  SECTION("Testing min at different positions")
  {
    arr[0] = 1.;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[0] = 3.9;

    arr[size - 1] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[size - 1] = 3.9;

    arr[size / 2] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
    arr[size / 2] = 3.9;

    arr[size / 4] = 1.;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.first == 1.);
    cudaFree(d_arr);
  }

  SECTION("Testing max at different positions")
  {
    arr[0] = 4.1;
    double* d_arr = make_gpu_arr<double, size>(arr);
    MinMax res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[0] = 3.9;

    arr[size - 1] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size - 1] = 3.9;

    arr[size / 2] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
    arr[size / 2] = 3.9;

    arr[size / 4] = 4.1;
    d_arr = make_gpu_arr<double, size>(arr);
    res = min_max<double>(d_arr, size);
    CHECK(res.second == 4.1);
    cudaFree(d_arr);
  }
}
