#ifndef QUAD_UTIL_CUDA_CUSTOM_FUNCTIONS_CUH
#define QUAD_UTIL_CUDA_CUSTOM_FUNCTIONS_CUH

#include <iostream>
#include <cuda.h>
#include "cuda/pagani/quad/GPUquad/Sample.cuh"
#include "cuda/pagani/quad/util/cudaDebugUtil.h"

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




template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
    const unsigned int FULL_MASK = 0xffffffff;
	
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = max(__shfl_xor_sync(FULL_MASK, val, mask), val);
    }
      
    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMin(T val)
{
    const unsigned int FULL_MASK = 0xffffffff;
	
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
        val = min(__shfl_xor_sync(FULL_MASK, val, mask), val);
		
    }
      
    return val;
}

template <typename T>
__device__ void
blockReduceMinMax(T& min, T& max)
{
    static __shared__ T shared_max[32];
	static __shared__ T shared_min[32];
	
    int lane = threadIdx.x % 32;   // 32 is for warp size
    int wid = threadIdx.x >> 5 /* threadIdx.x / 32  */;

    min = warpReduceMin(min);
	max = warpReduceMax(max);
	
    if (lane == 0) {
      shared_min[wid] = min;
	  shared_max[wid] = max;
	  //printf("all warps blockReduceMinMax [%i](%i) min:%f\n", blockIdx.x, threadIdx.x, min);
    }
    __syncthreads(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    min = (threadIdx.x < (blockDim.x >> 5)) ? shared_min[lane] : DBL_MAX;
	max = (threadIdx.x < (blockDim.x >> 5)) ? shared_max[lane] : 0.;

	
    if (wid == 0){
		min = warpReduceMin(min);
		max = warpReduceMax(max);
	}
}

template<typename T>
__global__ 
void 
blocks_min_max(const T* __restrict__ input, const int size, T* min, T* max)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int total_num_threads = blockDim.x * gridDim.x;
	
    T localMax = 0.f;
    T localMin = DBL_MAX;
	
	for (size_t i = tid; i < size; i += total_num_threads){
        T val = input[tid];
		
        if (localMax < val){
            localMax = val;
        }
		
		if(localMin > val){
			localMin = val;
			//printf("new min for [%i](%i):%f\n", blockIdx.x, threadIdx.x, localMin);
		}
    }
	
    blockReduceMinMax(localMin, localMax);  
	
    if (threadIdx.x == 0){
        max[blockIdx.x] = localMax;
		min[blockIdx.x] = localMin;
		//printf("block reduction stage [%i] :%f,%f\n", blockIdx.x, localMax, localMin);
    }
}

template<typename T>
__global__ void block0_min_max(T* mins, T* maxs, const int size, T* min, T* max)
{
	const int tid = threadIdx.x;
	
	
    T localMax = tid < size ? maxs[tid] : 0.;
    T localMin = tid < size ? mins[tid] : DBL_MAX;
	
    blockReduceMinMax(localMin, localMax);  
	
    if (threadIdx.x == 0){
        max[blockIdx.x] = localMax;
		min[blockIdx.x] = localMin;
		//printf("reducing the block results [%i] :%f,%f\n", blockIdx.x, localMax, localMin);
    }
}

template<typename T>
std::pair<double,double>
min_max(T* input, const int size){
	size_t num_threads = 1024;
	size_t max_num_blocks = 1024;
	size_t num_blocks = min((size + num_threads - 1)/num_threads, max_num_blocks);
	
	double* block_mins = cuda_malloc<double>(num_blocks);
	double* block_maxs = cuda_malloc<double>(num_blocks);
	double* d_min = cuda_malloc<double>(1);
	double* d_max = cuda_malloc<double>(1);
	
	blocks_min_max<double><<<num_blocks, num_threads>>>(input, size, block_mins, block_maxs);
	block0_min_max<double><<<1, std::max(num_blocks, (size_t)32)>>>(block_mins, block_maxs, num_blocks, d_min, d_max);
	
	cudaDeviceSynchronize();
	
	double min = 0.;
	double max = 0.;
	
	cuda_memcpy_to_host(&min, d_min, 1);
	cuda_memcpy_to_host(&max, d_max, 1);
	
	cudaFree(block_mins);
	cudaFree(block_maxs);
	cudaFree(d_min);
	cudaFree(d_max);
	return {min, max};
}

#endif