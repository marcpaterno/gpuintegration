#ifndef VEGAS_UTIL_UTIL_CUH
#define VEGAS_UTIL_UTIL_CUH

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
  
namespace VEGAS_namespace{  
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
}

#endif
