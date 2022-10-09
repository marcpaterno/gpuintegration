#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/quad/GPUquad/Interp2D.cuh"
#include "cuda/pagani/quad/util/cudaMemoryUtil.h"
#include "cuda/pagani/quad/GPUquad/Pagani.cuh"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

template<typename T>
__global__ void
gEvaluate(quad::Interp2D<T> f, T x, T y, T* result)
{
  *result = f(x, y);
}

template<typename T>
T
Evaluate(quad::Interp2D<T> f, T x, T y)
{
  T* result = quad::cuda_malloc_managed<T>(1);
  gEvaluate<<<1, 1>>>(f, x, y, result);
  cudaDeviceSynchronize();
  T hResult = *result;
  cudaFree(result);
  return hResult;
}

template<typename T>
__global__ void
gClamp(quad::Interp2D<T> f, T x, T y, T* result)
{
  *result = f.clamp(x, y);
}

template<typename T>
T
clamp(quad::Interp2D<T> f, T x, T y)
{
  T* result = quad::cuda_malloc_managed<T>(1);
  gClamp<T><<<1, 1>>>(f, 2.5, 4.5, result);
  cudaDeviceSynchronize();
  T hResult = *result;
  cudaFree(result);
  return hResult;
}

template<typename T>
void test_clamp_interface(){
  constexpr std::size_t nx = 3; // rows
  constexpr std::size_t ny = 2; // cols
  std::array<T, nx> xs = {1., 2., 3.};
  std::array<T, ny> ys = {4., 5.};
  std::array<T, ny * nx> zs;

  auto fxy = [](T x, T y) { return 3 * x * y + 2 * x + 4 * y; };

  for (std::size_t i = 0; i != nx; ++i) {
    T x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      T y = ys[j];
      zs[j * nx + i] = fxy(x, y);
    }
  }

  quad::Interp2D<T> f(xs, ys, zs);

  SECTION("interpolation works")
  {
    T x = 2.5;
    T y = 4.5;
    T true_result = 56.75;
    T interpResult = Evaluate<T>(f, x, y);
    CHECK(interpResult == true_result);
  }

  SECTION("extrapolation gets clamped")
  {
    T clampRes = clamp<T>(f, 0., 4.5);
    T interpResult = clamp<T>(f, 1., 4.5);
    CHECK(clampRes == interpResult); // to the left

    clampRes = clamp<T>(f, 4., 4.5);
    interpResult = clamp<T>(f, 3., 4.5);
    CHECK(clampRes == interpResult);

    clampRes = clamp<T>(f, 2., 3.);
    interpResult = clamp<T>(f, 2., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp<T>(f, 2., 5.5);
    interpResult = clamp<T>(f, 2., 5.);
    CHECK(clampRes == interpResult);

    clampRes = clamp<T>(f, 0., 0.);
    interpResult = clamp<T>(f, 1., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp<T>(f, 4., 3.);
    interpResult = clamp<T>(f, 3., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp<T>(f, 0., 6.);
    interpResult = clamp<T>(f, 1., 5.);
    CHECK(clampRes == interpResult);

    clampRes = clamp<T>(f, 4., 6.);
    interpResult = clamp<T>(f, 3., 5.);
    CHECK(clampRes == interpResult);
  }
}

template<typename T>
void test_interpolation_at_knots(){
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 2;
  std::array<T, nx> const xs = {1., 2., 3.};
  std::array<T, ny> const ys = {4., 5.};
  auto fxy = [](T x, T y) { return 3 * x * y + 2 * x + 4 * y; };
  std::array<T, ny * nx> zs;

  for (std::size_t i = 0; i != nx; ++i) {
    T x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      T y = ys[j];
      zs[j * nx + i] = fxy(x, y);
    }
  }

  quad::Interp2D<T> f(xs, ys, zs);
  for (std::size_t i = 0; i != nx; ++i) {
    T x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      T y = ys[j];
      CHECK(zs[j * nx + i] == fxy(x, y));
      T interpResult = Evaluate<T>(f, x, y);
      CHECK(zs[j * nx + i] == interpResult);
    }
  }
}

template<typename T>
void test_on_bilinear(){
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 4;
  std::array<T, nx> const xs = {1., 2., 3.};
  std::array<T, ny> const ys = {1., 2., 3., 4.};
  std::array<T, ny * nx> zs;

  auto fxy = [](T x, T y) { return 2 * x + 3 * y - 5; };

  for (std::size_t i = 0; i != nx; ++i) {
    T x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      T y = ys[j];
      zs[j * nx + i] = fxy(x, y);
      CHECK(zs[j * nx + i] == fxy(x, y));
    }
  }

<<<<<<< HEAD
  quad::Interp2D f(xs, ys, zs);

  double interpResult = Evaluate(f, 2.5, 1.5);
=======
  quad::Interp2D<T> f(xs, ys, zs);
  using IntegType = quad::Interp2D<T>;
  
  T interpResult = Evaluate<T>(f, 2.5, 1.5);
>>>>>>> 7a11a52 (flaoting point template argument wip, added on Interpolation Objects)
  CHECK(interpResult == 4.5);
}

<<<<<<< HEAD
  constexpr int ndim = 2;
  quad::Pagani<double, ndim> pagani;
  std::array<double, ndim> lows = {1., 1.};
  std::array<double, ndim> highs = {3., 4.};
  quad::Volume<double, ndim> vol(lows, highs);

  size_t free_physmem, total_physmem;

  for (int run = 0; run < 1; ++run) {
    cudaMemGetInfo(&free_physmem, &total_physmem);
    std::cout << "start"
              << "," << run << "," << free_physmem << std::endl;
    pagani.integrate<quad::Interp2D>(f, 1.e-3, 1.e-12, &vol);
    std::cout << "end"
              << "," << run << "," << free_physmem << std::endl;
  }
}
=======

TEST_CASE("clamp interface works"){
	test_clamp_interface<double>();
	test_clamp_interface<float>();
}

TEST_CASE("Interp2D exact at knots"){
	test_interpolation_at_knots<double>();
	test_interpolation_at_knots<float>();
}

TEST_CASE("Interp2D on bilinear"){
	test_on_bilinear<double>();
	test_on_bilinear<float>();
}

>>>>>>> 7a11a52 (flaoting point template argument wip, added on Interpolation Objects)
