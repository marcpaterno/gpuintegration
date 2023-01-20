#ifndef CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH
#define CUDACUHRE_QUAD_UTIL_CUDAARRAY_CUH

#include <cstring>
#include "cuda/pagani/quad/quad.h"

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
    __host__ __device__
    cudaDynamicArray(const cudaDynamicArray& a)
    {
#ifndef __CUDA_ARCH__
      N = a.N;
      cudaMallocManaged((void**)&data, sizeof(T) * a.N);
      memcpy(data, a.data, sizeof(T) * a.N);
#else
      // can't instantiate on device and then access on host
      N = a.N;
      data = new T[a.N];
      memcpy(data, a.data, sizeof(T) * a.N);
#endif
    }

    __host__ __device__
    cudaDynamicArray()
    {
      data = nullptr;
      N = 0;
    }

    // make everything host device

    cudaDynamicArray(T const* initData, size_t s)
    {
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

    explicit cudaDynamicArray(size_t s)
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

    /*cudaDynamicArray&
    operator=(const cudaDynamicArray& source)
    {
      cudaMallocManaged((void**)&data, sizeof(T) * source.size());
      cudaMemcpy(data, source.data, sizeof(T) * N, cudaMemcpyHostToDevice);
      return *this;
    }*/

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

};

#endif
