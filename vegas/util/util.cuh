#ifndef UTIL_CUH
#define UTIL_CUH

#include <cstring>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
  
  struct IntegrationResult{
	double estimate = 0;
	double errorest = 0;
	double chi_sq = 0.;  
	int status = 1;
  };

  template <class T>
  T*
  cuda_malloc_managed(size_t size)
  {
    T* temp = nullptr;
    auto rc = cudaMallocManaged(&temp, sizeof(T)*size);
    if (rc != cudaSuccess)
      throw std::bad_alloc();
    return temp;
  }

  template <class T>
  T*
  cuda_malloc_managed()
  {
    T* temp = nullptr;
    auto rc = cudaMallocManaged(&temp, sizeof(T));
    if (rc != cudaSuccess)
      throw std::bad_alloc();
    return temp;
  }

  template <class T>
  T*
  cuda_copy_to_managed(T const& on_host)
  {
    T* buffer = cuda_malloc_managed<T>();
    try {
      new (buffer) T(on_host);
    }
    catch (...) {
      cudaFree(buffer);
      throw;
    }
    return buffer;
  }
  
  /*
  namespace gpu {
  template <typename T, std::size_t s>
  class cudaArray {
  public:
    void
    Initialize(T const* initData)
    {
      std::memcpy(data, initData, sizeof(T) * s);
    }

    __host__ __device__ const T*
    begin() const
    {
      return &data[0];
    }

    __host__ __device__ const T*
    end() const
    {
      return (&data[0] + s);
    }

    __host__ __device__ constexpr std::size_t
    size() const
    {
      return s;
    }

    __host__ __device__ T&
    operator[](std::size_t i)
    {
      return data[i];
    }

    __host__ __device__ T const&
    operator[](std::size_t i) const
    {
      return data[i];
    }

    T data[s];
  };

  template <typename T>
  class cudaDynamicArray {
  public:
    cudaDynamicArray()
    {
      data = nullptr;
      N = 0;
    }

    // host-only function
    void
    Initialize(T const* initData, size_t s)
    {
      N = s;
      cudaMallocManaged((void**)&data, sizeof(T) * s);
      cudaMemcpy(data, initData, sizeof(T) * s, cudaMemcpyHostToDevice);
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

    cudaDynamicArray&
    operator=(const cudaDynamicArray& source)
    {
      cudaMallocManaged((void**)&data, sizeof(T) * source.size());
      cudaMemcpy(data, source.data, sizeof(T) * N, cudaMemcpyHostToDevice);
      return *this;
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
  }; // cudaDynamicArray

};*/
  
  
class Managed {
public:
  void*
  operator new(size_t len)
  {
    void* ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void
  operator delete(void* ptr)
  {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

/*
namespace gpu {

  namespace detail {
    template <class F, size_t N, std::size_t... I>
    __device__ double
    apply_impl(F&& f,
               gpu::cudaArray<double, N> const& data,
               std::index_sequence<I...>)
    {
      return f(data[I]...);
    };
  }

  template <class F, size_t N>
  __device__ double
  // Unsure if we need to pass 'f' by value, for GPU execution
  apply(F&& f, gpu::cudaArray<double, N> const& data)
  {
    return detail::apply_impl(
      std::forward<F>(f), data, std::make_index_sequence<N>());
  }
}*/

#endif
