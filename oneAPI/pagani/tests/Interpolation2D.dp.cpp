#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "catch2/catch.hpp"
#include "oneAPI/pagani/quad/GPUquad/Interp2D.hpp"
#include "oneAPI/pagani/quad/util/cudaMemoryUtil.h"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

void
gEvaluate(quad::Interp2D* f, double x, double y, double* result)
{
  *result = f->operator()(x, y);
}

double
Evaluate(quad::Interp2D* f, double x, double y)
{
  auto q_ct1 = sycl::queue(sycl::gpu_selector());;
  double* result = quad::cuda_malloc_managed<double>(1);
    q_ct1.parallel_for(
      sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
          gEvaluate(f, x, y, result);
      });
  q_ct1.wait_and_throw();
  double hResult = *result;
  sycl::free(result, q_ct1);
  return hResult;
}

void
gClamp(quad::Interp2D* f, double x, double y, double* result)
{
  *result = f->clamp(x, y);
}

double
clamp(quad::Interp2D* f, double x, double y)
{
  auto q_ct1 = sycl::queue(sycl::gpu_selector());;
  double* result = quad::cuda_malloc_managed<double>(1);
    q_ct1.parallel_for(
      sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
          gClamp(f, 2.5, 4.5, result);
      });
  q_ct1.wait_and_throw();
  double hResult = *result;
  sycl::free(result, q_ct1);
  return hResult;
}

TEST_CASE("clamp interface works")
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
  quad::Interp2D* d_f = cuda_copy_to_managed(f);
  
  SECTION("interpolation works")
  {
    double x = 2.5;
    double y = 4.5;
    double true_result = 56.75;
    double interpResult = Evaluate(d_f, x, y);
    CHECK(interpResult == true_result);
  }

  SECTION("extrapolation gets clamped")
  {
    double clampRes = clamp(d_f, 0., 4.5);
    double interpResult = clamp(d_f, 1., 4.5);
    CHECK(clampRes == interpResult); // to the left

    clampRes = clamp(d_f, 4., 4.5);
    interpResult = clamp(d_f, 3., 4.5);
    CHECK(clampRes == interpResult);

    clampRes = clamp(d_f, 2., 3.);
    interpResult = clamp(d_f, 2., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(d_f, 2., 5.5);
    interpResult = clamp(d_f, 2., 5.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(d_f, 0., 0.);
    interpResult = clamp(d_f, 1., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(d_f, 4., 3.);
    interpResult = clamp(d_f, 3., 4.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(d_f, 0., 6.);
    interpResult = clamp(d_f, 1., 5.);
    CHECK(clampRes == interpResult);

    clampRes = clamp(d_f, 4., 6.);
    interpResult = clamp(d_f, 3., 5.);
    CHECK(clampRes == interpResult);
  }
}

TEST_CASE("Interp2D exact at knots", "[interpolation][2d]")
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
  quad::Interp2D* d_f = cuda_copy_to_managed(f);
  
  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys[j];
      CHECK(zs[j * nx + i] == fxy(x, y));
      double interpResult = Evaluate(d_f, x, y);
      CHECK(zs[j * nx + i] == interpResult);
    }
  }
}

TEST_CASE("Interp2D on bilinear")
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
  quad::Interp2D* d_f = cuda_copy_to_managed(f);
  
  double interpResult = Evaluate(d_f, 2.5, 1.5);
  CHECK(interpResult == 4.5);
}


