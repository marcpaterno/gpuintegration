#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "common/cuda/Interp3D.cuh"
#include "common/cuda/cudaMemoryUtil.h"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

__global__ void
gEvaluate(quad::Interp3D f, double x, double y, double z, double* result)
{
  *result = f(x, y, z);
}

double
Evaluate(quad::Interp3D f, double x, double y, double z)
{
  double* result = quad::cuda_malloc_managed<double>(1);
  gEvaluate<<<1, 1>>>(f, x, y, z, result);
  cudaDeviceSynchronize();
  double hResult = *result;
  cudaFree(result);
  return hResult;
}

__global__ void
gClamp(quad::Interp3D f, double x, double y, double z, double* result)
{
  *result = f.clamp(x, y, z);
}

double
clamp(quad::Interp3D f, double x, double y, double z)
{
  double* result = quad::cuda_malloc_managed<double>(1);
  gClamp<<<1, 1>>>(f, x, y, z, result);
  cudaDeviceSynchronize();
  double hResult = *result;
  cudaFree(result);
  return hResult;
}

struct Indexer {
  Indexer(size_t xs, size_t ys, size_t zs) : _xs(xs), _ys(ys), _zs(zs) {}

  size_t
  operator()(double x, double y, double z)
  {
    return x + (_xs * y) + (_xs * _ys * z);
  }

private:
  size_t _xs = 0;
  size_t _ys = 0;
  size_t _zs = 0;
};

void
test_on_quadratic_func()
{
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 2;
  constexpr std::size_t nz = 4;

  std::array<double, nx> xs = {1., 2., 3.};
  std::array<double, ny> ys = {4., 5.};
  std::array<double, nz> zs = {4., 5., 6., 7.};

  std::array<double, ny * nx * nz> vs;
  Indexer index(nx, ny, nz);
  auto fxyz = [](double x, double y, double z) {
    return 2 * pow(x, 2) + 3 * pow(y, 2) + 4 * pow(z, 2) + 5 * x + 6 * y +
           7 * z + x * y - x * z + z * y;
  };

  for (std::size_t k = 0; k != nz; ++k) {
    double z = zs[k];
    for (std::size_t i = 0; i != nx; ++i) {
      double x = xs[i];
      for (std::size_t j = 0; j != ny; ++j) {
        double y = ys[j];
        vs[index(i, j, k)] = fxyz(x, y, z);
      }
    }
  }

  quad::Interp3D f(xs, ys, zs, vs);
  const double x = 2.1;
  const double y = 4.9;
  const double z = 6.1;
  double true_result = fxyz(x, y, z);
  double interpResult = Evaluate(f, x, y, z);
  CHECK(interpResult == Approx(true_result).epsilon(1.e-2));
}

void
test_on_linear_func()
{
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 2;
  constexpr std::size_t nz = 4;

  std::array<double, nx> xs = {1., 2., 3.};
  std::array<double, ny> ys = {4., 5.};
  std::array<double, nz> zs = {4., 5., 6., 7.};

  std::array<double, ny * nx * nz> vs;
  Indexer index(nx, ny, nz);
  auto fxyz = [](double x, double y, double z) {
    return 2 * x + 3 * y + 4 * z;
  };

  for (std::size_t k = 0; k != nz; ++k) {
    double z = zs[k];
    for (std::size_t i = 0; i != nx; ++i) {
      double x = xs[i];
      for (std::size_t j = 0; j != ny; ++j) {
        double y = ys[j];
        vs[index(i, j, k)] = fxyz(x, y, z);
      }
    }
  }

  quad::Interp3D f(xs, ys, zs, vs);

  SECTION("interpolation works for linear function")
  {
    const double x = 2.3;
    const double y = 4.3;
    const double z = 6.3;
    double true_result = fxyz(x, y, z);
    double interpResult = Evaluate(f, x, y, z);
    CHECK(interpResult == Approx(true_result).epsilon(1.e-15));
  }

  SECTION("extrapolation gets clamped")
  {
    double clampRes = clamp(f, 0., 4.5, 6.3);
    double interpResult = clamp(f, 1., 4.5, 6.3);
    CHECK(clampRes == interpResult); // to the left x

    clampRes = clamp(f, 4., 4.5, 6.3); // to the right x
    interpResult = clamp(f, 3., 4.5, 6.3);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 2., 3., 6.3); // to the left y
    interpResult = clamp(f, 2., 4., 6.3);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 2., 5.5, 6.3); // to the right y
    interpResult = clamp(f, 2., 5., 6.3);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 1.1, 4.2, 3.1); // to the left z
    interpResult = clamp(f, 1.1, 4.2, 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 1.1, 4.2, 7.4); // to the right z
    interpResult = clamp(f, 1.1, 4.2, 7.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 0., 3.1, 3.1);
    interpResult = clamp(f, 1., 4., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(f, 3.5, 5.6, 7.3);
    interpResult = clamp(f, 3., 5., 7.);
    CHECK(clampRes == interpResult);
  }
}

void
linear_interpolation_at_knots()
{
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 2;
  constexpr std::size_t nz = 4;

  std::array<double, nx> xs = {1., 2., 3.};
  std::array<double, ny> ys = {4., 5.};
  std::array<double, nz> zs = {4., 5., 6., 7.};

  std::array<double, ny * nx * nz> vs;
  Indexer index(nx, ny, nz);
  auto fxyz = [](double x, double y, double z) {
    return 2 * x + 3 * y + 4 * z;
  };

  for (std::size_t k = 0; k != nz; ++k) {
    double z = zs[k];
    for (std::size_t i = 0; i != nx; ++i) {
      double x = xs[i];
      for (std::size_t j = 0; j != ny; ++j) {
        double y = ys[j];
        vs[index(i, j, k)] = fxyz(x, y, z);
      }
    }
  }

  quad::Interp3D f(xs, ys, zs, vs);
  for (std::size_t k = 0; k != nz; ++k) {
    double z = zs[k];
    for (std::size_t i = 0; i != nx; ++i) {
      double x = xs[i];
      for (std::size_t j = 0; j != ny; ++j) {
        double y = ys[j];
        CHECK(Evaluate(f, x, y, z) == fxyz(x, y, z));
      }
    }
  }
}

void
quadratic_interpolation_at_knots()
{
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 2;
  constexpr std::size_t nz = 4;

  std::array<double, nx> xs = {1., 2., 3.};
  std::array<double, ny> ys = {4., 5.};
  std::array<double, nz> zs = {4., 5., 6., 7.};

  std::array<double, ny * nx * nz> vs;
  Indexer index(nx, ny, nz);
  auto fxyz = [](double x, double y, double z) {
    return 2 * pow(x, 2) + 3 * pow(y, 2) + 4 * pow(z, 2) + 5 * x + 6 * y +
           7 * z + x * y - x * z + z * y;
  };

  for (std::size_t k = 0; k != nz; ++k) {
    double z = zs[k];
    for (std::size_t i = 0; i != nx; ++i) {
      double x = xs[i];
      for (std::size_t j = 0; j != ny; ++j) {
        double y = ys[j];
        vs[index(i, j, k)] = fxyz(x, y, z);
      }
    }
  }

  quad::Interp3D f(xs, ys, zs, vs);
  for (std::size_t k = 0; k != nz; ++k) {
    double z = zs[k];
    for (std::size_t i = 0; i != nx; ++i) {
      double x = xs[i];
      for (std::size_t j = 0; j != ny; ++j) {
        double y = ys[j];
        CHECK(Evaluate(f, x, y, z) == fxyz(x, y, z));
      }
    }
  }
}

TEST_CASE("interpolation at knots for linear function")
{
  linear_interpolation_at_knots();
}

TEST_CASE("interpolation at knots for quadratic function")
{
  quadratic_interpolation_at_knots();
}

TEST_CASE("exact solution on linear function")
{
  test_on_linear_func();
}

TEST_CASE("approximate solution on quadratic function")
{
  test_on_quadratic_func();
}