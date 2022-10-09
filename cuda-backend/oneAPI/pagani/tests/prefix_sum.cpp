#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include <iostream>
#include "oneAPI/pagani/quad/GPUquad/PaganiUtils.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaMemoryUtil.h"
#include "oneAPI/pagani/quad/util/thrust_utils.dp.hpp"
#include "oneAPI/pagani/quad/util/custom_functions.dp.hpp"


TEST_CASE("Exclusvie scan of array of size 8")
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
	constexpr size_t size = 8;
	std::array<int, size> arr = {3, 1, 7, 0, 4, 1, 6, 3};
	std::array<int, size> true_results = {0, 3, 4, 11, 11, 15, 16, 22};
	std::array<int, size> custom_results;
    std::array<int, size> thrust_results;

	int *out = quad::cuda_malloc<int>(size);
	int *d_arr = quad::cuda_malloc<int>(size);
    
	cuda_memcpy_to_device<int>(d_arr, arr.data(), size);
	
	sum_scan_blelloch(out, d_arr, size);
	cuda_memcpy_to_host<int>(custom_results.data(), out, size);
    
	SECTION("Check results of custom function")
	{
		for(int i = 0; i < size; ++i)
			CHECK(true_results[i] == custom_results[i]);
	}
	
    
    sycl::free(out, q_ct1);
    out = quad::cuda_malloc<int>(size);
    
	thrust_exclusive_scan<int>(d_arr, size, out);
    cuda_memcpy_to_host<int>(thrust_results.data(), out, size);
    
	SECTION("Thrust Gets the same results")
	{
		for(int i = 0; i < size; ++i){
            CHECK(true_results[i] == thrust_results[i]);
        }
	}
	
	
	sycl::free(d_arr, q_ct1);
	sycl::free(out, q_ct1);
}


TEST_CASE("Exclusvie scan of array of non-power-two size")
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
	constexpr size_t size = 10000;
	std::array<int, size> arr;
	std::iota(arr.begin(), arr.end(), 1.);
	std::array<int, size> h_out_thrust;
    std::array<int, size> h_out_custom;
    
	int *out_thrust = quad::cuda_malloc<int>(size);
	int *out_custom = quad::cuda_malloc<int>(size);
	int *d_arr = quad::cuda_malloc<int>(size);
	cuda_memcpy_to_device<int>(d_arr, arr.data(), size);
	
	sum_scan_blelloch(out_custom, d_arr, size);
	thrust_exclusive_scan<int>(d_arr, size, out_thrust);
    
	cuda_memcpy_to_host<int>(h_out_thrust.data(), out_thrust, size);
    cuda_memcpy_to_host<int>(h_out_custom.data(), out_custom, size);
    
	SECTION("Check results of custom function")
	{
		for(int i = 0; i < size; ++i)
			CHECK(h_out_custom[i] == h_out_thrust[i]);
	}
	
	sycl::free(d_arr, q_ct1);
	sycl::free(out_thrust, q_ct1);
	sycl::free(out_custom, q_ct1);
}


TEST_CASE("Exclusvie scan of array of odd size")
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
	constexpr size_t size = 10001;
	std::array<int, size> arr;
	std::iota(arr.begin(), arr.end(), 1.);
	std::array<int, size> h_out_thrust;
    std::array<int, size> h_out_custom;
    
    
	int *out_thrust = quad::cuda_malloc<int>(size);
	int *out_custom = quad::cuda_malloc<int>(size);
	int *d_arr = quad::cuda_malloc<int>(size);
	cuda_memcpy_to_device<int>(d_arr, arr.data(), size);
	
	sum_scan_blelloch(out_custom, d_arr, size);
	thrust_exclusive_scan<int>(d_arr, size, out_thrust);
	cuda_memcpy_to_host<int>(h_out_thrust.data(), out_thrust, size);
    cuda_memcpy_to_host<int>(h_out_custom.data(), out_custom, size);
    
	SECTION("Check results of custom function")
	{
		for(int i = 0; i < size; ++i)
			CHECK(h_out_thrust[i] == h_out_custom[i]);
	}
	
	sycl::free(d_arr, q_ct1);
	sycl::free(out_thrust, q_ct1);
	sycl::free(out_custom, q_ct1);
}


TEST_CASE("Exclusvie scan of array of size 8 double type")
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
	constexpr size_t size = 8;
	std::array<double, size> arr = {3., 1., 7., 0., 4., 1., 6., 3.};
	std::array<double, size> true_results = {0., 3., 4., 11., 11., 15., 16., 22.};
	std::array<double, size> custom_results;
    std::array<double, size> thrust_results;

	double *out = quad::cuda_malloc<double>(size);
	double *d_arr = quad::cuda_malloc<double>(size);
    
	cuda_memcpy_to_device<double>(d_arr, arr.data(), size);
	
	sum_scan_blelloch(out, d_arr, size);
	cuda_memcpy_to_host<double>(custom_results.data(), out, size);
    
	SECTION("Check results of custom function")
	{
		for(int i = 0; i < size; ++i)
			CHECK(true_results[i] == custom_results[i]);
	}
	
    
    sycl::free(out, q_ct1);
    out = quad::cuda_malloc<double>(size);
    
	thrust_exclusive_scan<double>(d_arr, size, out);
    cuda_memcpy_to_host<double>(thrust_results.data(), out, size);
    
	SECTION("Thrust Gets the same results")
	{
		for(int i = 0; i < size; ++i){
            CHECK(true_results[i] == thrust_results[i]);
        }
	}
	
	
	sycl::free(d_arr, q_ct1);
	sycl::free(out, q_ct1);
}
