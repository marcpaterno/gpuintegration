#define CATCH_CONFIG_MAIN

#include "catch2/catch.hpp"
#include "cuda/cudaPagani/quad/util/cudaArray.cuh"
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/universal_vector.h>

namespace noCopyConstr{
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

    cudaDynamicArray(T const* initData, size_t s) { 
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

    __host__ __device__ T& operator[](std::size_t i) { return data[i]; }

    __host__ __device__ T operator[](std::size_t i) const { return data[i]; }

    T* data;
    size_t N;
  }; 
    
}

template<typename arrayType>
__global__ void
Evaluate(arrayType input)
{
    //make device to device copy and write to copied
    arrayType input2(input);
    input2[1] = 9.3;
    input[1] = 99.;
}

template<typename arrayType>
__global__ void
Evaluate_pass_by_ref(arrayType& input)
{
    input[1] = 99.;
}

__global__ void
evaluate_thrust(double* vec, size_t size)
{
    //for(auto& v: vec)
    //    v = 99.;
    for(int i=0; i <size; ++i)
        vec[i] = 99.;
}


TEST_CASE("cudaDynamicArray Data Access")
{
    size_t arraySize = 5;
    gpu::cudaDynamicArray<double> array;
    array.Reserve(arraySize);
    
    //initialize data allocated in unified memory
    for(int i=0; i<arraySize; ++i)
        array[i] = i;
    
    Evaluate<gpu::cudaDynamicArray<double>><<<1,1>>>(array);
    cudaDeviceSynchronize();
    
    SECTION("No shallow copy when copying device to device")
    {
        CHECK(array[1] != 9.3); //redundant but makes point clearer
    }
    
    SECTION("Deep copy from host to device")
    {
        CHECK(array[1] == 1.);
    }
    
    SECTION("Default copy-constructor makes shallow copy")
    {
        noCopyConstr::cudaDynamicArray<double> input;
        input.Reserve(arraySize);
         
        for(int i=0; i<arraySize; ++i)
            input[i] = i;
        
        Evaluate<noCopyConstr::cudaDynamicArray<double>><<<1,1>>>(input);
        cudaDeviceSynchronize();
        
        CHECK(input[1] == 99.);
    }
}

/*TEST_CASE("unified vector test")
{
    thrust::universal_vector<double> vec;
    CHECK(vec.size() == 0);
    vec.push_back(4.0);
    CHECK(vec.size() == 1);
    
    evaluate_thrust<<<1,1>>>(vec.data(), vec.size());
    cudaDeviceSynchronize();
    for(auto const& v: vec)
        CHECK(v == 99.);
}*/