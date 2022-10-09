#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/quad/GPUquad/Interp1D.cuh"
#include "cuda/pagani/quad/util/cudaMemoryUtil.h"

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

template<typename T>
__global__ void
Evaluate(quad::Interp1D<T> interpolator,
         size_t size,
         T* input,
         T* results)
{
  for (size_t i = 0; i < size; i++) {
    results[i] = interpolator(input[i]);
  }
}

template<typename T>
__global__ void
Evaluate(quad::Interp1D<T> interpolator, T value, T* result)
{
  *result = interpolator(value);
}

template<typename T>
void
interpolate_at_knots(){
	const size_t s = 9;
	std::array<T, s> xs = {1., 2., 3., 4., 5., 6, 7., 8., 9.};
	std::array<T, s> ys = xs;

	auto Transform = [](std::array<T, s>& ys) {
		for (T& elem : ys)
			elem = 2 * elem * (3 - elem) * std::cos(elem);
	};
	
	Transform(ys);
	quad::Interp1D<T> interpObj(xs, ys);

	T* input = quad::cuda_malloc_managed<T>(s);
	for (size_t i = 0; i < s; i++)
		input[i] = xs[i];

	T* results = quad::cuda_malloc_managed<T>(s);

	Evaluate<T><<<1, 1>>>(interpObj, s, input, results);
	cudaDeviceSynchronize();

	for (std::size_t i = 0; i < s; ++i) {
		CHECK(ys[i] == results[i]);
	}
	cudaFree(results);
}

template<typename T>
void
interpolate_on_quadratic(){
	const size_t s = 5;
	std::array<T, s> xs = {1., 2., 3., 4., 5.};
	std::array<T, s> ys = xs;

	auto Transform = [](std::array<T, s>& ys) {
		for (auto& elem : ys)
		elem = elem * elem;
	};
	Transform(ys);
	quad::Interp1D<T> interpObj(xs, ys);

	T* result = quad::cuda_malloc_managed<T>(1);
	T interp_point = 1.41421;
	T true_interp_res = 2.24263;
	Evaluate<T><<<1, 1>>>(interpObj, interp_point, result);
	cudaDeviceSynchronize();
	CHECK(*result == Approx(true_interp_res).epsilon(1e-4));
  
	interp_point = 2.41421;
	true_interp_res = 6.07105;
	Evaluate<T><<<1, 1>>>(interpObj, interp_point, result);
	cudaDeviceSynchronize();
	CHECK(*result == Approx(true_interp_res).epsilon(1e-4));
  
	interp_point = 3.41421;
	true_interp_res = 11.89947;
	Evaluate<T><<<1, 1>>>(interpObj, interp_point, result);
	cudaDeviceSynchronize();
	CHECK(*result == Approx(true_interp_res).epsilon(1e-4));
	
	interp_point = 4.41421;
	true_interp_res = 19.72789;
	Evaluate<T><<<1, 1>>>(interpObj, interp_point, result);
	cudaDeviceSynchronize();
	CHECK(*result == Approx(true_interp_res).epsilon(1e-4));

	cudaFree(result);
}

TEST_CASE("Interp1D exact at knots", "[interpolation][1d]")
{
	interpolate_at_knots<double>();
	interpolate_at_knots<float>();
}

TEST_CASE(){
	interpolate_on_quadratic<double>();
	interpolate_on_quadratic<float>();
}
