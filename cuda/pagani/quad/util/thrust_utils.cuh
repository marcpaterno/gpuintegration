#ifndef QUAD_THRUST_UTILS_CUH
#define QUAD_THRUST_UTILS_CUH

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include "cuda/pagani/quad/util/custom_functions.cuh"

template<typename T1, typename T2, bool use_custom = false>
double
dot_product(T1* arr1, T2* arr2, const size_t size){
	
	if constexpr(use_custom == false){
		thrust::device_ptr<T1> wrapped_mask_1  = thrust::device_pointer_cast(arr1);
		thrust::device_ptr<T2> wrapped_mask_2 = thrust::device_pointer_cast(arr2);
		double res = thrust::inner_product(thrust::device,
									  wrapped_mask_2,
									  wrapped_mask_2 + size,
									  wrapped_mask_1,
									  0.);
		
		return res;
	}
	else{
		double res = custom_inner_product_atomics<T1, T2>(arr1, arr2, size);
		return res;
	}
}

template<typename T, bool use_custom =  false>
T
reduction(T* arr, size_t size){
	if constexpr(use_custom == false){
		thrust::device_ptr<T> wrapped_ptr = thrust::device_pointer_cast(arr);
		return thrust::reduce(wrapped_ptr, wrapped_ptr + size);
	}
	else{
		return custom_reduce_atomics(arr, size);
	}
}


#endif