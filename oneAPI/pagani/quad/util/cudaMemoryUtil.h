#ifndef CUDACUHRE_QUAD_UTIL_CUDAMEMORY_UTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDAMEMORY_UTIL_H

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "oneAPI/pagani/quad/util/cudaDebugUtil.h"

namespace quad {

  template <typename T>
  T*
  copy_to_host(T* device_arr, size_t size)
  {
    T* host_arr = new T[size];
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    q_ct1.memcpy(host_arr, device_arr, sizeof(T) * size)
      .wait();
    return host_arr;
  }

  size_t
  GetAmountFreeMem()
  {
    size_t free_physmem, total_physmem;
    
    return 16e9;

    
  }

  class Managed {
  public:
    void*
    operator new(size_t len)
    {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      void* ptr;
      ptr = (void*)sycl::malloc_shared(len, q_ct1);
      return ptr;
    }

    void
    operator delete(void* ptr)
    {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      sycl::free(ptr, q_ct1);
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
    try {
      size_t free_physmem, total_physmem;
            return (double)free_physmem / total_physmem;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    size_t
    GetAmountFreeMem()
    try {
      size_t free_physmem, total_physmem;
      
      
      return 0;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    int
    AllocateMemory(void** d_ptr, size_t n)
    try {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      return (*d_ptr = (void*)sycl::malloc_device(n, q_ct1),
              0);
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    int
    AllocateUnifiedMemory(void** d_ptr, size_t n)
    try {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      return (*d_ptr = (void*)sycl::malloc_shared(n, q_ct1),
              0);
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    int
    ReleaseMemory(void* d_ptr)
    try {
      auto q_ct1 =  sycl::queue(sycl::gpu_selector());
      return (sycl::free(d_ptr, q_ct1), 0);
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    int
    SetHeapSize(size_t hSize = (size_t)2 * 1024 * 1024 * 1024)
     {

      
      return 0;
    }

    //@brief Initialize Device
    int
      DeviceInit(int dev = -1, int verbose = 0){}
    
  };

  template <class T>
  T*
  cuda_malloc_managed(size_t size)
  {
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    CudaCheckError();
    T* temp = nullptr;
    temp = sycl::malloc_shared<T>(size, q_ct1);
    return temp;
  }

  template <class T>
  T*
  cuda_malloc_managed()
  {
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    T* temp = nullptr;
    temp = sycl::malloc_shared<T>(1, q_ct1);
    return temp;
  }

  template <class T>
  T*
  cuda_copy_to_managed(T const& on_host)
  {
    T* buffer = cuda_malloc_managed<T>();
    new (buffer) T(on_host);
    return buffer;
  }

  template <class T>
  T*
  cuda_malloc(size_t size)
  {
    T* temp;
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    temp = sycl::malloc_device<T>(size, q_ct1);
    return temp;
  }

  template <typename T>
  void
  cuda_memcpy_to_device(T* dest, T* src, size_t size)
  {

    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    
    q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
  }

  template <typename T>
  void
  cuda_memcpy_to_device(T* dest, const T* src, size_t size)
  {
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
  }

  template <typename T>
  void
  cuda_memcpy_device_to_device(T* dest, const T* src, size_t size)
  {
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    
    q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
  }

  template <typename T>
  void
  cuda_memcpy_device_to_device(T* dest, T* src, size_t size)
  {
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
  }

  template <typename T>
  void
  cuda_memcpy_device_to_host(T* dest, T* src, size_t size)
  {
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
  }

  template <class T>
  T*
  cuda_copy_to_device(T const& on_host)
  {
    T* buffer = cuda_malloc<T>(1);
    const T* hp = &on_host;
    cuda_memcpy_to_device<T>(buffer, hp, 1);
    CudaCheckError();
  return buffer;
  }
}

#endif
