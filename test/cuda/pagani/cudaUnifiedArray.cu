#define CATCH_CONFIG_MAIN

#include "catch2/catch.hpp"
#include "common/cuda/cudaArray.cuh"


/*
namespace noCopyConstr {
  template <typename T>
  class cudaDynamicArray {
  public:
    cudaDynamicArray()
    {
#ifndef __CUDA_ARCH__
      printf("default constructor called from host side\n");
#else
      printf("default constructor called from device side\n");
#endif
      data = nullptr;
      N = 0;
    }

    cudaDynamicArray(T const* initData, size_t s)
    {
      printf("constructor with cudaMallocManaged called from host side\n");
      Initialize(initData, s);
    }

    void
    Initialize(T const* initData, size_t s)
    {
      N = s;
      cudaMallocManaged((void**)&data, sizeof(T) * s);
      cudaMemcpy(data, initData, sizeof(T) * s, cudaMemcpyHostToDevice);
    }

    void
    Reserve(size_t s)
    {
      N = s;
      cudaMallocManaged((void**)&data, sizeof(T) * s);
    }

    cudaDynamicArray(size_t s)
    {
      N = s;
      cudaMallocManaged((void**)&data, sizeof(T) * s);
    }
    __host__ __device__ ~cudaDynamicArray()
    {
#ifndef __CUDACC__
      cudaFree(data);
#endif
    }

    __host__ __device__ const T*
    begin() const
    {
      return &data[0];
    }

    __host__ __device__ const T*
    end() const
    {
      return (&data[0] + N);
    }

    __host__ __device__ constexpr std::size_t
    size() const
    {
      return N;
    }

    __host__ __device__ T&
    operator[](std::size_t i)
    {
      return data[i];
    }

    __host__ __device__ T
    operator[](std::size_t i) const
    {
      return data[i];
    }

    T* data;
    size_t N;
  };

}
*/
template <typename arrayType>
__global__ void
set_vals_at_indices(arrayType array, arrayType indices, arrayType vals)
{
  for(int i=0; i < indices.size(); ++i){
	const size_t index_to_change = indices[i];
	array[index_to_change] = vals[i];
  }
}

template <typename arrayType, typename T>
__global__ void
set_vals_at_indices(T* array, arrayType indices, arrayType vals)
{
  for(int i=0; i < indices.size(); ++i){
	const size_t index_to_change = indices[i];
	array[index_to_change] = vals[i];
  }
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
	std::array<int, vals_to_edit> indices = {1, 3, 4};
	std::array<int, vals_to_edit> vals = {11, 33, 44};
	
	int_array d_indices(indices.data(), indices.size());
	int_array d_vals(vals.data(), vals.size());

	SECTION("c-style array constructor works"){
		CHECK(d_indices[0] == 1);
		CHECK(d_indices[1] == 3);
		CHECK(d_indices[2] == 4);
		
		CHECK(d_vals[0] == 11);
		CHECK(d_vals[1] == 33);
		CHECK(d_vals[2] == 44);
	}
	
	set_vals_at_indices<int_array><<<1,1>>>(array, d_indices, d_vals);
	cudaDeviceSynchronize();
	
	SECTION("Copy constructor makes deep-copy"){
		//passing by value to kernel invokes copy-constructor, which does deep, not shallow copy
		//thus values don't update when accessing the array on host
		CHECK(array[1] != 11);
		CHECK(array[3] != 33);
		CHECK(array[4] != 44);
	}
	
	set_vals_at_indices<int_array, int><<<1,1>>>(array.data(), d_indices, d_vals);
	cudaDeviceSynchronize();
	
	SECTION("Can still access data on host after editing on device"){
		//if we pass pointer to that data (which is allocated in unified memory)
		//we can get update on the device properly
		CHECK(array[1] == 11);
		CHECK(array[3] == 33);
		CHECK(array[4] == 44);
	}
	
	int_array copy(array);
	SECTION("copy-constructor works"){
		for(int i=0; i < array.size(); ++i)
			CHECK(copy[i] == array[i]);
	}
}
