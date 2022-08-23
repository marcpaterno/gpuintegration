#ifndef CUDACUHRE_QUAD_UTIL_CUDAMEMORY_UTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDAMEMORY_UTIL_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuda/pagani/quad/util/cudaDebugUtil.h"

namespace quad {
	
	template<typename T>
	T*
	copy_to_host(T* device_arr, size_t size){
		T* host_arr = new T[size];
		dpct::get_default_queue().memcpy(host_arr, device_arr, sizeof(T) * size).wait();
		return host_arr;
	}
	
	size_t
      GetAmountFreeMem()
     try {
      size_t free_physmem, total_physmem;
      /*
      DPCT1003:44: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      /*
      DPCT1072:45: DPC++ currently does not support getting the available memory
      on the current device. You may need to adjust the code.
      */
      QuadDebugExit(
        (total_physmem =
           dpct::get_current_device().get_device_info().get_global_mem_size(),
         0));
      return free_physmem;
    }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  class Managed {
  public:
    void *operator new(size_t len) {
      void *ptr;
      ptr = (void*)sycl::malloc_shared(len, dpct::get_default_queue());
      dpct::get_current_device().queues_wait_and_throw();
      return ptr;
    }

    void operator delete(void *ptr) {
      dpct::get_current_device().queues_wait_and_throw();
      sycl::free(ptr, dpct::get_default_queue());
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
      /*
      DPCT1003:46: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      /*
      DPCT1072:47: DPC++ currently does not support getting the available memory
      on the current device. You may need to adjust the code.
      */
      QuadDebugExit(
        (total_physmem =
           dpct::get_current_device().get_device_info().get_global_mem_size(),
         0));
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
      /*
      DPCT1003:48: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      /*
      DPCT1072:49: DPC++ currently does not support getting the available memory
      on the current device. You may need to adjust the code.
      */
      QuadDebugExit(
        (total_physmem =
           dpct::get_current_device().get_device_info().get_global_mem_size(),
         0));
      return free_physmem;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    int
    AllocateMemory(void** d_ptr, size_t n)
     try {
      /*
      DPCT1003:50: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      return (*d_ptr = (void*)sycl::malloc_device(n, dpct::get_default_queue()),
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
      /*
      DPCT1003:51: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      return (*d_ptr = (void*)sycl::malloc_shared(n, dpct::get_default_queue()),
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
      /*
      DPCT1003:52: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      return (sycl::free(d_ptr, dpct::get_default_queue()), 0);
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    int
    SetHeapSize(size_t hSize = (size_t)2 * 1024 * 1024 * 1024)
     try {
      /*
      DPCT1027:53: The call to cudaDeviceSetLimit was replaced with 0 because
      DPC++ currently does not support setting resource limits on devices.
      */
      return 0;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
	
    //@brief Initialize Device
    int
    DeviceInit(int dev = -1, int verbose = 0)
     try {
      int error = 0;

      do {
        int deviceCount;
        /*
        DPCT1003:62: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        error = QuadDebug(
          (deviceCount = dpct::dev_mgr::instance().device_count(), 0));
        /*
        DPCT1000:57: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (error)
          /*
          DPCT1001:56: The statement could not be removed.
          */
          break;
        if (deviceCount == 0) {
          fprintf(stderr, "No devices supporting CUDA.\n");
          exit(1);
        }

        if ((dev > deviceCount - 1) || (dev < 0)) {
          dev = 0;
        }

        //error = QuadDebug(cudaSetDevice(dev));
        /*
        DPCT1000:59: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (error)
          /*
          DPCT1001:58: The statement could not be removed.
          */
          break;

        size_t free_physmem, total_physmem;
        /*
        DPCT1003:63: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1072:64: DPC++ currently does not support getting the available
        memory on the current device. You may need to adjust the code.
        */
        QuadDebugExit(
          (total_physmem =
             dpct::get_current_device().get_device_info().get_global_mem_size(),
           0));

        dpct::device_info deviceProp;
        /*
        DPCT1003:65: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        error = QuadDebug((
          dpct::dev_mgr::instance().get_device(dev).get_device_info(deviceProp),
          0));
        /*
        DPCT1000:61: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (error)
          /*
          DPCT1001:60: The statement could not be removed.
          */
          break;

        /*
        DPCT1005:66: The SYCL device version is different from CUDA Compute
        Compatibility. You may need to rewrite this code.
        */
        if (deviceProp.get_major_version() < 1) {
          fprintf(stderr, "Device does not support CUDA.\n");
          exit(1);
        }
        if (false && verbose) {
          printf("Using device %d: %s (SM%d, %d SMs, %lld free / %lld total MB "
                 "physmem, ECC %s)\n",
                 dev,
                 deviceProp.get_name(),
                 /*
                 DPCT1005:67: The SYCL device version is different from CUDA
                 Compute Compatibility. You may need to rewrite this code.
                 */
                 deviceProp.get_major_version() * 100 +
                   deviceProp.get_minor_version() * 10,
                 deviceProp.get_max_compute_units(),
                 (unsigned long long)free_physmem / 1024 / 1024,
                 (unsigned long long)total_physmem / 1024 / 1024,
                 (dpct::get_current_device()
                    .get_info<sycl::info::device::error_correction_support>()) ?
                   "on" :
                   "off");
          fflush(stdout);
        }

      } while (0);
      return error;
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
  };

  template <class T>
    T*
    cuda_malloc_managed(size_t size)
    {
      CudaCheckError();
      T* temp = nullptr;
      /*
      DPCT1003:68: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      auto rc = (temp = (T*)sycl::malloc_shared<T>(size,
                                                dpct::get_default_queue()),
                 0);
      if (rc != 0) {
        temp = nullptr;

        /*
        DPCT1072:69: DPC++ currently does not support getting the available
        memory on the current device. You may need to adjust the code.
        */
      
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
      /*
      DPCT1003:70: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      auto rc =
        (temp = (T*)sycl::malloc_shared<T>(1, dpct::get_default_queue()),
         0);
      if (rc != 0) {
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
        sycl::free(buffer, dpct::get_default_queue());
        throw;
      }
      return buffer;
    }

  template <class T>
  T*
  cuda_malloc(size_t size) try {
    T* temp;
    /*
    DPCT1003:74: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    auto rc = (temp = (T*)sycl::malloc_device<T>(size,
                                              dpct::get_default_queue()),
               0);
    /*
    DPCT1000:73: Error handling if-stmt was detected but could not be rewritten.
    */
    if (rc != 0) {
      /*
      DPCT1001:72: The statement could not be removed.
      */
      throw std::bad_alloc();
    }
    return temp;
  }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

  template<typename T>
    void
    cuda_memcpy_to_device(T* dest, T* src, size_t size){
    /*
    DPCT1003:75: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    auto rc =
      (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
    if (rc != 0) {
      printf("error in cuda_mempcy_to_device with host src\n");
      throw std::bad_alloc();
      abort();
    }
  }

  template<typename T>
    void
    cuda_memcpy_to_device(T* dest, const T* src, size_t size){
    /*
    DPCT1003:76: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    auto rc =
      (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
    if (rc != 0) {
      printf("error in cuda_mempcy_to_device with host src\n");
      CudaCheckError();
      throw std::bad_alloc();
    }
  }

  template<typename T>
    void
    cuda_memcpy_device_to_device(T* dest, const T* src, size_t size){
    /*
    DPCT1003:77: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    auto rc =
      (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
    if (rc != 0) {
      printf("error in cuda_memcpy_device_to_device const src\n");
      throw std::bad_alloc();
    }
  }

  template<typename T>
    void
    cuda_memcpy_device_to_device(T* dest, T* src, size_t size){
    /*
    DPCT1003:78: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    auto rc =
      (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
    if (rc != 0) {
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
     try {
      size_t free_physmem, total_physmem;
      /*
      DPCT1003:79: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      /*
      DPCT1072:80: DPC++ currently does not support getting the available memory
      on the current device. You may need to adjust the code.
      */
      QuadDebugExit(
        (total_physmem =
           dpct::get_current_device().get_device_info().get_global_mem_size(),
         0));
      return free_physmem;
    }
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }
}

#endif
