#ifndef QUAD_UTIL_CUDA_CUSTOM_FUNCTIONS_CUH
#define QUAD_UTIL_CUDA_CUSTOM_FUNCTIONS_CUH

#include <iostream>
#include <cuda.h>
#include "cuda/pagani/quad/GPUquad/Sample.cuh"

/*
	require blocks to be equal to size
*/

template<typename T>
__global__ 
void
device_custom_reduce(T* arr, size_t size, T* out){
    T sum = 0.;
	//reduce multiple elements per thread
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int total_num_threads = blockDim.x * gridDim.x;
	
	for (size_t i = tid; i < size; i += total_num_threads) {
		sum += arr[i];
	}
	
	sum = quad::blockReduceSum(sum);
	
	if (threadIdx.x == 0)
		out[blockIdx.x] = sum;
}

template<typename T>
T
custom_reduce(T* arr, size_t size){
	size_t num_threads = 512;
	size_t max_num_blocks = 1024;
	size_t num_blocks = min((size + num_threads - 1)/num_threads, max_num_blocks);
	T* out = cuda_malloc<T>(num_blocks);
	
	device_custom_reduce<<<num_blocks, num_threads>>>(arr, size, out);
	device_custom_reduce<<<1, 1024>>>(out, num_blocks, out);
	cudaFree(out);
}

template<typename T>
__global__ 
void
device_custom_reduce_atomics(T* arr, size_t size, T* out){
    T sum = 0.;
	//reduce multiple elements per thread
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int total_num_threads = blockDim.x * gridDim.x;
	
	for (size_t i = tid; i < size; i += total_num_threads) {
		sum += arr[i];
	}
	
	sum = quad::warpReduceSum(sum);
	const int warpSize = 32;
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd(out, sum);
	}
}

template<typename T>
T
custom_reduce_atomics(T* arr, size_t size){
	T res = 0.;
	size_t num_threads = 512;
	size_t max_num_blocks = 1024;
	size_t num_blocks = min((size + num_threads - 1)/num_threads, max_num_blocks);
	T* out = cuda_malloc<T>(1);
	cuda_memcpy_to_device<T>(out, &res, 1);
	
	cuda_memcpy_to_device<T>(out, &res, 1);
	device_custom_reduce_atomics<<<num_blocks, num_threads>>>(arr, size, out);
	
	cuda_memcpy_to_host<T>(&res, out, 1);
	cudaFree(out);
	return res;
}

template<typename T1, typename T2>
__global__ 
void
device_custom_inner_product_atomics(T1* arr1, T2* arr2, size_t size, T2* out){
    T2 sum = 0.;
	//reduce multiple elements per thread
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int total_num_threads = blockDim.x * gridDim.x;
	
	for (size_t i = tid; i < size; i += total_num_threads) {
		T2 temp = arr1[i]*arr2[i];
		sum += arr1[i]*arr2[i];
	}
	
	sum = quad::warpReduceSum(sum);
	const int warpSize = 32;
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd(out, sum);
	}
}

template<typename T1, typename T2>
T2
custom_inner_product_atomics(T1* arr1, T2* arr2, size_t size){
	T2 res = 0.;
	size_t num_threads = 512;
	size_t max_num_blocks = 1024;
	size_t num_blocks = min((size + num_threads - 1)/num_threads, max_num_blocks);
	T2* out = cuda_malloc<T2>(1);
	cuda_memcpy_to_device<T2>(out, &res, 1);	
	device_custom_inner_product_atomics<T1, T2><<<num_blocks, num_threads>>>(arr1, arr2, size, out);
	cuda_memcpy_to_host<T2>(&res, out, 1);
	cudaFree(out);
	return res;
}

#endif