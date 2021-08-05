#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cudaPagani/quad/GPUquad/Interp1D.cuh"
#include "cudaPagani/quad/util/cudaMemoryUtil.h"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

__global__ void
Evaluate(quad::Interp1D interpolator,
         size_t size,
         double* input,
         double* results)
{
  for (size_t i = 0; i < size; i++) {
    results[i] = interpolator(input[i]);
  }
}

__global__ void
Evaluate(quad::Interp1D interpolator, double value, double* result)
{
  *result = interpolator(value);
}

TEST_CASE("Interp1D exact at knots", "[interpolation][1d]")
{
  // user's input arrays
  const size_t s = 9;
  std::array<double, s> xs = {1., 2., 3., 4., 5., 6, 7., 8., 9.};
  std::array<double, s> ys = xs;

  auto Transform = [](std::array<double, s>& ys) {
    for (double& elem : ys)
      elem = 2 * elem * (3 - elem) * std::cos(elem);
  };
  Transform(ys);
  quad::Interp1D interpObj(xs, ys);

  double* input = quad::cuda_malloc_managed<double>(s);
  for (size_t i = 0; i < s; i++)
    input[i] = xs[i];

  double* results = quad::cuda_malloc_managed<double>(s);

  Evaluate<<<1, 1>>>(interpObj, s, input, results);
  cudaDeviceSynchronize();

  for (std::size_t i = 0; i < s; ++i) {
    CHECK(ys[i] == results[i]);
  }
  cudaFree(results);
}

TEST_CASE("Interp1D on quadratic")
{
  const size_t s = 3;
  std::array<double, s> xs = {1., 2., 3.};
  std::array<double, s> ys = xs;

  auto Transform = [](std::array<double, s>& ys) {
    for (auto& elem : ys)
      elem = elem * elem;
  };
  Transform(ys);
  quad::Interp1D interpObj(xs, ys);

  double* result = quad::cuda_malloc_managed<double>(1);

  Evaluate<<<1, 1>>>(interpObj, 1.41421, result);
  cudaDeviceSynchronize();

  CHECK(*result == Approx(2.24263).epsilon(1e-4));
  cudaFree(result);
}
