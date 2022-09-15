#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/catch2/catch.hpp"
#include "oneAPI/pagani/quad/GPUquad/Interp1D.hpp"
#include "oneAPI/pagani/quad/util/cudaMemoryUtil.h"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

void
Evaluate(quad::Interp1D* interpolator,
         size_t size,
         double* input,
         double* results)
{
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range(size), [=](sycl::item<1> item_ct1) {
			results[item_ct1] = interpolator->operator()(input[item_ct1]);
        });
    }).wait();
}

void
Evaluate(quad::Interp1D* interpolator, double value, double* result)
{
	dpct::get_default_queue().submit([&](sycl::handler& cgh) {
           
        cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1),
                                            sycl::range(1, 1, 1)),
                             [=](sycl::nd_item<3> item_ct1) {
            *result = interpolator->operator()(value);
        });
    }).wait();
}

TEST_CASE("Interp1D exact at knots", "[interpolation][1d]")
{
  using Interp1D = quad::Interp1D;
  // user's input arrays
  const size_t s = 9;
  std::array<double, s> xs = {1., 2., 3., 4., 5., 6, 7., 8., 9.};
  std::array<double, s> ys = xs;

  auto Transform = [](std::array<double, s>& ys) {
    for (double& elem : ys)
      elem = 2 * elem * (3 - elem) * std::cos(elem);
  };
  Transform(ys);
  Interp1D interpObj(xs, ys);
  Interp1D* d_interpObj = cuda_copy_to_managed(interpObj);
  
  double* input = quad::cuda_malloc_managed<double>(s);
  for (size_t i = 0; i < s; i++)
    input[i] = xs[i];

  double* results = quad::cuda_malloc_managed<double>(s);

  Evaluate(d_interpObj, s, input, results);

  for (std::size_t i = 0; i < s; ++i) {
    CHECK(ys[i] == results[i]);
  }
  
  sycl::free(results, dpct::get_default_queue());
  sycl::free(input, dpct::get_default_queue());
  d_interpObj->~Interp1D();
  sycl::free(d_interpObj, dpct::get_default_queue());

}

TEST_CASE("Interp1D on quadratic")
{
  using Interp1D = quad::Interp1D;
  const size_t s = 5;
  std::array<double, s> xs = {1., 2., 3., 4., 5.};
  std::array<double, s> ys = xs;

  auto Transform = [](std::array<double, s>& ys) {
    for (auto& elem : ys)
      elem = elem * elem;
  };
  
  Transform(ys);
  Interp1D interpObj(xs, ys);
  Interp1D* d_interpObj = cuda_copy_to_managed(interpObj);
  
  double* result = quad::cuda_malloc_managed<double>(1);
  double interp_point = 1.41421;
  double true_interp_res = 2.24263;
  Evaluate(d_interpObj, interp_point, result);
  CHECK(*result == Approx(true_interp_res).epsilon(1e-4));
  
  interp_point = 2.41421;
  true_interp_res = 6.07105;
  Evaluate(d_interpObj, interp_point, result);
  CHECK(*result == Approx(true_interp_res).epsilon(1e-4));
  
  interp_point = 3.41421;
  true_interp_res = 11.89947;
  Evaluate(d_interpObj, interp_point, result);
  CHECK(*result == Approx(true_interp_res).epsilon(1e-4));
	
  interp_point = 4.41421;
  true_interp_res = 19.72789;
  Evaluate(d_interpObj, interp_point, result);
  CHECK(*result == Approx(true_interp_res).epsilon(1e-4));
  
  d_interpObj->~Interp1D();
  sycl::free(d_interpObj, dpct::get_default_queue());
  sycl::free(result, dpct::get_default_queue());
}
