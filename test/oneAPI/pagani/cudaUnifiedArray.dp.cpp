#define CATCH_CONFIG_MAIN

#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "catch2/catch.hpp"
#include "common/oneAPI/cudaArray.dp.hpp"
#include "common/oneAPI/cudaMemoryUtil.h"
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/universal_vector.h>


template <typename arrayType, typename T>
void
set_vals_at_indices(T* array, arrayType* indices, arrayType* vals)
{
	sycl::queue q;
	q.submit([&](sycl::handler& cgh) {
		cgh.parallel_for(
		  sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
		  [=](sycl::nd_item<3> item_ct1) {
			  
			for(int i=0; i < (*indices).size(); ++i){
				const size_t index_to_change = (*indices)[i];
				array[index_to_change] = (*vals)[i];
			}
		});
	}).wait();
}
	  
TEST_CASE("Data can be set on the device and accessed on host")
{
	using int_array = gpu::cudaDynamicArray<int>;
	int_array array;
	array.Reserve(5);
	
	for (int i = 0; i < array.size(); ++i)
		array[i] = i;
	
	SECTION("Data can be set and accessed on host"){
		CHECK(array[0] == 0);
		CHECK(array[4] == 4);
	}
	
	constexpr int vals_to_edit = 3;
	std::array<int, vals_to_edit> vals = {11, 33, 44};
	std::array<int, vals_to_edit> indices = {1, 3, 4};
	
	int_array d_vals(vals.data(), vals.size());
	int_array d_indices(indices.data(), indices.size());
	
	SECTION("copy-constructor with c-style array works"){
		CHECK(d_vals[0] == 11);
		CHECK(d_vals[1] == 33);
		CHECK(d_vals[2] == 44);
		
		CHECK(d_indices[0] == 1);
		CHECK(d_indices[1] == 3);
		CHECK(d_indices[2] == 4);
	}
	
	
	//create pointer accessible on device memory
	int_array* d_vals_ptr = quad::cuda_copy_to_managed<int_array>(d_vals);
	int_array* d_indices_ptr = quad::cuda_copy_to_managed<int_array>(d_indices);	

	SECTION("copy-constructor on managed memory works"){		
		CHECK((*d_indices_ptr)[0] == 1);
		CHECK((*d_indices_ptr)[1] == 3);
		CHECK((*d_indices_ptr)[2] == 4);
		
		CHECK((*d_vals_ptr)[0] == 11);
		CHECK((*d_vals_ptr)[1] == 33);
		CHECK((*d_vals_ptr)[2] == 44);
		
		
	}
		
	//MISSING kernel that passes array by value due to cudaDynamicArray not being device-copyable
	//that's why we need to pass a pointer, that points to managed memory
	set_vals_at_indices<int_array, int>(array.data(), d_indices_ptr, d_vals_ptr);
	
	SECTION("Can still access data on host after editing on device"){
		//if we pass pointer to that data (which is allocated in unified memory)
		//we can get update on the device properly
		CHECK(array[1] == 11);
		CHECK(array[3] == 33);
		CHECK(array[4] == 44);
	}
}
