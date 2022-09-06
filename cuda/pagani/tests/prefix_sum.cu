#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/demos/function.cuh"
#include "cuda/pagani/quad/GPUquad/Pagani.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
#include "cuda/pagani/quad/util/mem_util.cuh"
#include "cuda/pagani/quad/util/cudaMemoryUtil.h"
#include "cuda/pagani/quad/GPUquad/Sub_region_filter.cuh"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <array>



TEST_CASE("Exclusvie scan of array of size 8")
{
	constexpr size_t size = 8;
	std::array<int, size> arr = {3, 1, 7, 0, 4, 1, 6, 3};
	std::array<int, size> true_results = {0, 3, 4, 11, 11, 15, 16, 22};
	
	int *out = quad::cuda_malloc_managed<int>(size);
	int *d_arr = quad::cuda_malloc_managed<int>(size);
	cuda_memcpy_to_device<int>(d_arr, arr.data(), size);
	
	sum_scan_blelloch(out, d_arr, size);
	
	//for(int i = 0; i < size; ++i)
	//	std::cout<<out[i]<<std::endl;
	
	SECTION("Check results of custom function")
	{
		for(int i = 0; i < size; ++i)
			CHECK(true_results[i] == out[i]);
	}
	
	for(int i=0; i < size; ++i){
		out[i] = 0;	
	}
	
	thrust_exclusive_scan<int>(d_arr, size, out);

	
	SECTION("Thrust Gets the same results")
	{
		for(int i = 0; i < size; ++i)
			CHECK(true_results[i] == out[i]);
	}
	
	
	cudaFree(d_arr);
	cudaFree(out);
}

TEST_CASE("Exclusvie scan of array of non-power-two size")
{
	constexpr size_t size = 10000;
	std::array<int, size> arr;
	std::iota(arr.begin(), arr.end(), 1.);
	
	int *out_thrust = quad::cuda_malloc_managed<int>(size);
	int *out_custom = quad::cuda_malloc_managed<int>(size);
	int *d_arr = quad::cuda_malloc_managed<int>(size);
	cuda_memcpy_to_device<int>(d_arr, arr.data(), size);
	
	sum_scan_blelloch(out_custom, d_arr, size);
	thrust_exclusive_scan<int>(d_arr, size, out_thrust);
	
	SECTION("Check results of custom function")
	{
		for(int i = 0; i < size; ++i)
			CHECK(out_thrust[i] == out_custom[i]);
	}
	
	cudaFree(d_arr);
	cudaFree(out_thrust);
	cudaFree(out_custom);
}


TEST_CASE("Exclusvie scan of array of odd size")
{
	constexpr size_t size = 10001;
	std::array<int, size> arr;
	std::iota(arr.begin(), arr.end(), 1.);
	
	int *out_thrust = quad::cuda_malloc_managed<int>(size);
	int *out_custom = quad::cuda_malloc_managed<int>(size);
	int *d_arr = quad::cuda_malloc_managed<int>(size);
	cuda_memcpy_to_device<int>(d_arr, arr.data(), size);
	
	sum_scan_blelloch(out_custom, d_arr, size);
	thrust_exclusive_scan<int>(d_arr, size, out_thrust);
	
	SECTION("Check results of custom function")
	{
		for(int i = 0; i < size; ++i)
			CHECK(out_thrust[i] == out_custom[i]);
	}
	
	cudaFree(d_arr);
	cudaFree(out_thrust);
	cudaFree(out_custom);
}