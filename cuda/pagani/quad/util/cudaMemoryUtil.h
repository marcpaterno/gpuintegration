#ifndef CUDACUHRE_QUAD_UTIL_CUDAMEMORY_UTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDAMEMORY_UTIL_H

#include "cuda/pagani/quad/util/cudaDebugUtil.h"
#include <cuda.h>

namespace quad {
	
  size_t
      GetAmountFreeMem()
    {
      size_t free_physmem, total_physmem;
      QuadDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));
      return free_physmem;
    }	
	
  class Managed {
  public:
    void *operator new(size_t len) {
      void *ptr;
      cudaMallocManaged(&ptr, len);
      cudaDeviceSynchronize();
      return ptr;
    }

    void operator delete(void *ptr) {
      cudaDeviceSynchronize();
      cudaFree(ptr);
    }
  };
  
  template <typename T>
    class MemoryUtil {};

  template <typename T>
    class HostMemory : public MemoryUtil<T> {
  public:
    void*
      AllocateMemory(void* ptr, size_t n)
    {
      ptr = malloc(n);
      return ptr;
    }

    void
      ReleaseMemory(void* ptr)
    {
      free(ptr);
    }
  };

  template <typename T>
    class DeviceMemory : public MemoryUtil<T> {
  public:
    double
      GetFreeMemPercentage()
    {
      size_t free_physmem, total_physmem;
      QuadDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));
      return (double)free_physmem / total_physmem;
    }

    size_t
      GetAmountFreeMem()
    {
      size_t free_physmem, total_physmem;
      QuadDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));
      return free_physmem;
    }

    cudaError_t
      AllocateMemory(void** d_ptr, size_t n)
    {
      return cudaMalloc(d_ptr, n);
    }

    cudaError_t
      AllocateUnifiedMemory(void** d_ptr, size_t n)
    {
      return cudaMallocManaged(d_ptr, n);
    }

    cudaError_t
      ReleaseMemory(void* d_ptr)
    {
      return cudaFree(d_ptr);
    }

    cudaError_t
      SetHeapSize(size_t hSize = (size_t)2 * 1024 * 1024 * 1024)
    {
      return cudaDeviceSetLimit(cudaLimitMallocHeapSize, hSize);
    }

    cudaError_t
      CopyHostToDeviceConstantMemory(const char* d_ptr, void* h_ptr, size_t n)
    {
      return cudaMemcpyToSymbol(d_ptr, h_ptr, n);
    }

    cudaError_t
      CopyHostToDeviceConstantMemory(const void* d_ptr, void* h_ptr, size_t n)
    {
      return cudaMemcpyToSymbol(d_ptr, h_ptr, n);
    }

    //@brief Initialize Device
    cudaError_t
      DeviceInit(int dev = -1, int verbose = 0)
    {
      cudaError_t error = cudaSuccess;

      do {
        int deviceCount;
        error = QuadDebug(cudaGetDeviceCount(&deviceCount));
        if (error)
          break;
        if (deviceCount == 0) {
          fprintf(stderr, "No devices supporting CUDA.\n");
          exit(1);
        }

        if ((dev > deviceCount - 1) || (dev < 0)) {
          dev = 0;
        }

        //error = QuadDebug(cudaSetDevice(dev));
        if (error)
          break;
        
        size_t free_physmem, total_physmem;
        QuadDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));

        cudaDeviceProp deviceProp;
        error = QuadDebug(cudaGetDeviceProperties(&deviceProp, dev));
        if (error)
          break;

        if (deviceProp.major < 1) {
          fprintf(stderr, "Device does not support CUDA.\n");
          exit(1);
        }
        if (false && verbose) {
          printf("Using device %d: %s (SM%d, %d SMs, %lld free / %lld total MB "
                 "physmem, ECC %s)\n",
                 dev,
                 deviceProp.name,
                 deviceProp.major * 100 + deviceProp.minor * 10,
                 deviceProp.multiProcessorCount,
                 (unsigned long long)free_physmem / 1024 / 1024,
                 (unsigned long long)total_physmem / 1024 / 1024,
                 (deviceProp.ECCEnabled) ? "on" : "off");
          fflush(stdout);
        }

      } while (0);
      return error;
    }
  };

  template <class T>
    T*
    cuda_malloc_managed(size_t size)
    {
      CudaCheckError();
      T* temp = nullptr;
      auto rc = cudaMallocManaged(&temp, sizeof(T) * size);
      if (rc != cudaSuccess){
		temp = nullptr;

		size_t free_physmem, total_physmem;
		cudaMemGetInfo(&free_physmem, &total_physmem);
		printf("cuda_malloc_managed(size) allocating size %lu free mem:%lu\n", size, free_physmem);
      
		CudaCheckError();
		throw std::bad_alloc();
      }
      return temp;
    }

  template <class T>
    T*
    cuda_malloc_managed()
    {
      T* temp = nullptr;
      auto rc = cudaMallocManaged(&temp, sizeof(T));
      if (rc != cudaSuccess){
	size_t free_physmem, total_physmem;
	cudaMemGetInfo(&free_physmem, &total_physmem);
	printf("cuda_malloc_managed() allocating size %lu free mem:%lu\n", sizeof(T), free_physmem);
	CudaCheckError();
	throw std::bad_alloc();
      }
      CudaCheckError();
      return temp;
    }

  template <class T>
    T*
    cuda_copy_to_managed(T const& on_host)
    {
      T* buffer = cuda_malloc_managed<T>();
      CudaCheckError();
      try {
	new (buffer) T(on_host);
	CudaCheckError();
      }
      catch (...) {
	cudaFree(buffer);
	throw;
      }
      return buffer;
    }
  
  template<class T>
    T*
    cuda_malloc(size_t size){
    T* temp;  
    auto rc = cudaMalloc((void**)&temp, sizeof(T) * size);
    if (rc != cudaSuccess){
      throw std::bad_alloc();
    }
    return temp;
  }

    


  template<typename T>
    void
    cuda_memcpy_to_device(T* dest, T* src, size_t size){
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
    if(rc != cudaSuccess){ 
      printf("error in cuda_mempcy_to_device with host src\n");
      throw std::bad_alloc();
      abort();
    }
  }

  template<typename T>
    void
    cuda_memcpy_to_device(T* dest, const T* src, size_t size){
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
    if(rc != cudaSuccess){
      printf("error in cuda_mempcy_to_device with host src\n");
      CudaCheckError();
      throw std::bad_alloc();
    }
  }

  template<typename T>
    void
    cuda_memcpy_device_to_device(T* dest, const T* src, size_t size){
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToDevice);
    if(rc != cudaSuccess){
      printf("error in cuda_memcpy_device_to_device const src\n");
      throw std::bad_alloc();
    }
  }

  template<typename T>
    void
    cuda_memcpy_device_to_device(T* dest, T* src, size_t size){
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToDevice);
    if(rc != cudaSuccess){
      printf("error in cuda_memcpy_device_to_device\n");
      throw std::bad_alloc();
      abort();
    }
  }

  template<class T>
    T*
    cuda_copy_to_device(T const& on_host){
      T* buffer = cuda_malloc<T>(1);
      const T* hp = &on_host;
      cuda_memcpy_to_device<T>(buffer, hp, 1);
      CudaCheckError();
      return buffer;
    }

  size_t
  get_free_mem()
    {
      size_t free_physmem, total_physmem;
      QuadDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));
      return free_physmem;
    }

}

#endif
