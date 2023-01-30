#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "common/cuda/Interp2D.cuh"
#include "common/cuda/cudaMemoryUtil.h"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

__global__ void
gEvaluate(quad::Interp2D f, double x, double y, double* result)
{
  *result = f(x, y);
}

double
Evaluate(quad::Interp2D f, double x, double y)
{
  double* result = quad::cuda_malloc_managed<double>(1);
  gEvaluate<<<1, 1>>>(f, x, y, result);
  cudaDeviceSynchronize();
  double hResult = *result;
  cudaFree(result);
  return hResult;
}

__global__ void
gClamp(quad::Interp2D f, double x, double y, double* result)
{
  *result = f.clamp(x, y);
}

double
clamp(quad::Interp2D f, double x, double y)
{
  double* result = quad::cuda_malloc_managed<double>(1);
  gClamp<<<1, 1>>>(f, 2.5, 4.5, result);
  cudaDeviceSynchronize();
  double hResult = *result;
  cudaFree(result);
  return hResult;
}

void
test_clamp_interface()
{
  constexpr std::size_t nx = 3; // rows
  constexpr std::size_t ny = 2; // cols
  std::array<double, nx> xs = {1., 2., 3.};
  std::array<double, ny> ys = {4., 5.};
  std::array<double, ny * nx> zs;

  auto fxy = [](double x, double y) { return 3 * x * y + 2 * x + 4 * y; };

  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys[j];
      zs[j * nx + i] = fxy(x, y);
    }
  }

  quad::Interp2D f(xs, ys, zs);

  SECTION("interpolation works")
  {
    double x = 2.5;
    double y = 4.5;
    double true_result = 56.75;
    double interpResult = Evaluate(f, x, y);
    CHECK(interpResult == true_result);
  }

  SECTION("extrapolation gets clamped")
  {
    double clampRes = clamp(f, 0., 4.5);
    double interpResult = clamp(f, 1., 4.5);
    CHECK(clampRes == interpResult); // to the left

    clampRes = clamp(f, 4., 4.5);
    interpResult = clamp(f, 3., 4.5);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 2., 3.);
    interpResult = clamp(f, 2., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 2., 5.5);
    interpResult = clamp(f, 2., 5.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 0., 0.);
    interpResult = clamp(f, 1., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 4., 3.);
    interpResult = clamp(f, 3., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 0., 6.);
    interpResult = clamp(f, 1., 5.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 4., 6.);
    interpResult = clamp(f, 3., 5.);
    CHECK(clampRes == interpResult);
  }
}

void
test_interpolation_at_knots()
{
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 2;
  std::array<double, nx> const xs = {1., 2., 3.};
  std::array<double, ny> const ys = {4., 5.};
  auto fxy = [](double x, double y) { return 3 * x * y + 2 * x + 4 * y; };
  std::array<double, ny * nx> zs;

  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys[j];
      zs[j * nx + i] = fxy(x, y);
    }
  }

  quad::Interp2D f(xs, ys, zs);
  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys[j];
      CHECK(zs[j * nx + i] == fxy(x, y));
      double interpResult = Evaluate(f, x, y);
      CHECK(zs[j * nx + i] == interpResult);
    }
  }
}

void
test_on_bilinear()
{
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 4;
  std::array<double, nx> const xs = {1., 2., 3.};
  std::array<double, ny> const ys = {1., 2., 3., 4.};
  std::array<double, ny * nx> zs;

  auto fxy = [](double x, double y) { return 2 * x + 3 * y - 5; };

  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys[j];
      zs[j * nx + i] = fxy(x, y);
      CHECK(zs[j * nx + i] == fxy(x, y));
    }
  }

  quad::Interp2D f(xs, ys, zs);
  using IntegType = quad::Interp2D;

  double interpResult = Evaluate(f, 2.5, 1.5);
  CHECK(interpResult == 4.5);
}

TEST_CASE("clamp interface works")
{
  test_clamp_interface();
}

TEST_CASE("Interp2D exact at knots")
{
  test_interpolation_at_knots();
}
