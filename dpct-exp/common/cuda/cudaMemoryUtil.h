#ifndef CUDACUHRE_QUAD_UTIL_CUDAMEMORY_UTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDAMEMORY_UTIL_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "dpct-exp/common/cuda/cudaDebugUtil.h"

namespace quad {

  size_t
  GetAmountFreeMem()
   try {
    size_t free_physmem, total_physmem;
    /*
    DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    /*
    DPCT1072:9: DPC++ currently does not support getting the available memory on
    the current device. You may need to adjust the code.
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
    void*
    operator new(size_t len)
    {
      void* ptr;
      ptr = (void*)sycl::malloc_shared(len, dpct::get_default_queue());
      dpct::get_current_device().queues_wait_and_throw();
      return ptr;
    }

    void
    operator delete(void* ptr)
    {
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
      DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      /*
      DPCT1072:11: DPC++ currently does not support getting the available memory
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
      DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      /*
      DPCT1072:13: DPC++ currently does not support getting the available memory
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
      DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted.
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
      DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted.
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
      DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted.
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
      DPCT1027:17: The call to cudaDeviceSetLimit was replaced with 0 because
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
        DPCT1003:26: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        error = QuadDebug(
          (deviceCount = dpct::dev_mgr::instance().device_count(), 0));
        /*
        DPCT1000:21: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (error)
          /*
          DPCT1001:20: The statement could not be removed.
          */
          break;
        if (deviceCount == 0) {
          fprintf(stderr, "No devices supporting CUDA.\n");
          exit(1);
        }

        if ((dev > deviceCount - 1) || (dev < 0)) {
          dev = 0;
        }

        // error = QuadDebug(cudaSetDevice(dev));
        /*
        DPCT1000:23: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (error)
          /*
          DPCT1001:22: The statement could not be removed.
          */
          break;

        size_t free_physmem, total_physmem;
        /*
        DPCT1003:27: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        /*
        DPCT1072:28: DPC++ currently does not support getting the available
        memory on the current device. You may need to adjust the code.
        */
        QuadDebugExit(
          (total_physmem =
             dpct::get_current_device().get_device_info().get_global_mem_size(),
           0));

        dpct::device_info deviceProp;
        /*
        DPCT1003:29: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        error = QuadDebug((
          dpct::dev_mgr::instance().get_device(dev).get_device_info(deviceProp),
          0));
        /*
        DPCT1000:25: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (error)
          /*
          DPCT1001:24: The statement could not be removed.
          */
          break;

        /*
        DPCT1005:30: The SYCL device version is different from CUDA Compute
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
                 DPCT1005:31: The SYCL device version is different from CUDA
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
    DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    auto rc = (temp = (T*)sycl::malloc_shared(sizeof(T) * size,
                                              dpct::get_default_queue()),
               0);
    if (rc != 0) {
      temp = nullptr;

      size_t free_physmem, total_physmem;
      /*
      DPCT1072:33: DPC++ currently does not support getting the available memory
      on the current device. You may need to adjust the code.
      */
      total_physmem =
        dpct::get_current_device().get_device_info().get_global_mem_size();
      printf("cuda_malloc_managed(size) allocating size %lu free mem:%lu\n",
             size,
             free_physmem);

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
    DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    auto rc =
      (temp = (T*)sycl::malloc_shared(sizeof(T), dpct::get_default_queue()), 0);
    if (rc != 0) {
      size_t free_physmem, total_physmem;
      /*
      DPCT1072:35: DPC++ currently does not support getting the available memory
      on the current device. You may need to adjust the code.
      */
      total_physmem =
        dpct::get_current_device().get_device_info().get_global_mem_size();
      printf("cuda_malloc_managed() allocating size %lu free mem:%lu\n",
             sizeof(T),
             free_physmem);
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
  cuda_malloc(size_t size)
   try {
    T* temp;
    /*
    DPCT1003:38: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    auto rc = (temp = (T*)sycl::malloc_device(sizeof(T) * size,
                                              dpct::get_default_queue()),
               0);
    /*
    DPCT1000:37: Error handling if-stmt was detected but could not be rewritten.
    */
    if (rc != 0) {
      /*
      DPCT1001:36: The statement could not be removed.
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

  template <typename T>
  void
  cuda_memcpy_to_device(T* dest, T* src, size_t size)
   try {
    /*
    DPCT1003:39: Migrated API does not return error code. (*, 0) is inserted.
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
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <typename T>
  void
  cuda_memcpy_to_device(T* dest, const T* src, size_t size)
   try {
    /*
    DPCT1003:40: Migrated API does not return error code. (*, 0) is inserted.
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
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  template <typename T>
  void
  cuda_memcpy_device_to_device(T* dest, T* src, size_t size)
  {
    /*
    DPCT1003:41: Migrated API does not return error code. (*, 0) is inserted.
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

  size_t
  get_free_mem()
   try {
    size_t free_physmem, total_physmem;
    /*
    DPCT1003:42: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    /*
    DPCT1072:43: DPC++ currently does not support getting the available memory
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

template <typename T>
T*
copy_to_host(T* src, size_t size)
{
  T* dest = new T[size];
  /*
  DPCT1003:44: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  auto rc =
    (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
  if (rc != 0)
    throw std::bad_alloc();
  return dest;
}

template <typename T>
void
cuda_memcpy_to_host(T* dest, T const* src, size_t n_elements)
 try {
  auto rc =
    /*
    DPCT1003:45: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * n_elements).wait(),
     0);
  if (rc != 0)
    throw std::bad_alloc();
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename T>
void
cuda_memcpy_device_to_device(T* dest, T const* src, size_t size)
{
  /*
  DPCT1003:46: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  auto rc =
    (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
  if (rc != 0)
    throw std::bad_alloc();
}

template <typename T>
struct Range {
  Range() = default;
  Range(T l, T h) : low(l), high(h) {}
  T low = 0., high = 0.;
};

template <typename T>
void
device_print_array(T* arr, size_t size)
{
  for (size_t i = 0; i < size; ++i)
    printf(
      "arr[%lu]:%i\n", i, arr[i]); // can't print arbitrary types from device,
                                   // must fix to do std::cout from host
}

template <typename T>
void
print_device_array(T* arr, size_t size)
{
    dpct::get_default_queue().parallel_for(
      sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        device_print_array<T>(arr, size);
      });
  dpct::get_current_device().queues_wait_and_throw();
}

template <class T>
T*
host_alloc(size_t size)
{
  T* temp = new T[size];
  if (temp == nullptr) {
	printf("cannot allocate on host\n");
    throw std::bad_alloc();
  }
  return temp;
}

// rename to  free_and_reallocate, delete copy_size, unecessary
template <typename T>
void
ExpandcuArray(T*& array, int currentSize, int newSize)
{
  int copy_size = std::min(currentSize, newSize);
  T* temp = cuda_malloc<T>(newSize);
  sycl::free(array, dpct::get_default_queue());
  array = temp;
}

template <typename IntegT>
IntegT*
make_gpu_integrand(const IntegT& integrand)
{
  IntegT* d_integrand;
  d_integrand =
    (IntegT*)sycl::malloc_shared(sizeof(IntegT), dpct::get_default_queue());
  memcpy(d_integrand, &integrand, sizeof(IntegT));
  return d_integrand;
}

template <typename T>
void
set_array_to_value(T* array, size_t size, T val, sycl::nd_item<3> item_ct1)
{
  size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
  if (tid < size) {
    array[tid] = val;
  }
}

template <typename T>
void
set_array_range_to_value(T* array,
                         size_t first_to_change,
                         size_t last_to_change,
                         size_t total_size,
                         T val,
                         sycl::nd_item<3> item_ct1)
{
  size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
  if (tid >= first_to_change && tid <= last_to_change && tid < total_size) {
    array[tid] = val;
  }
}

template <typename T>
void
set_device_array_range(T* arr,
                       size_t first_to_change,
                       size_t last_to_change,
                       size_t size,
                       T val)
{
  size_t num_threads = 64;
  size_t num_blocks = size / num_threads + ((size % num_threads) ? 1 : 0);
  /*
  DPCT1049:47: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
    dpct::get_default_queue().parallel_for(
      sycl::nd_range(sycl::range(1, 1, num_blocks) *
                       sycl::range(1, 1, num_threads),
                     sycl::range(1, 1, num_threads)),
      [=](sycl::nd_item<3> item_ct1) {
        set_array_range_to_value<T>(
          arr, first_to_change, last_to_change, size, val, item_ct1);
      });
  dpct::get_current_device().queues_wait_and_throw();
}

template <typename T>
void
set_device_array(T* arr, size_t size, T val)
{
  size_t num_threads = 64;
  size_t num_blocks = size / num_threads + ((size % num_threads) ? 1 : 0);
  /*
  DPCT1049:48: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
    dpct::get_default_queue().parallel_for(
      sycl::nd_range(sycl::range(1, 1, num_blocks) *
                       sycl::range(1, 1, num_threads),
                     sycl::range(1, 1, num_threads)),
      [=](sycl::nd_item<3> item_ct1) {
        set_array_to_value<T>(arr, size, val, item_ct1);
      });
  dpct::get_current_device().queues_wait_and_throw();
}

template <typename T, typename C = T>
bool
array_values_smaller_than_val(T* dev_arr, size_t dev_arr_size, C val)
{
  double* host_arr = host_alloc<double>(dev_arr_size);
  dpct::get_default_queue()
    .memcpy(host_arr, dev_arr, sizeof(double) * dev_arr_size)
    .wait();

  for (int i = 0; i < dev_arr_size; i++) {
    if (host_arr[i] >= static_cast<T>(val))
      return false;
  }
  return true;
}

template <typename T, typename C = T>
bool
array_values_larger_than_val(T* dev_arr, size_t dev_arr_size, C val)
{
  double* host_arr = host_alloc<double>(dev_arr_size);
  dpct::get_default_queue()
    .memcpy(host_arr, dev_arr, sizeof(double) * dev_arr_size)
    .wait();

  for (int i = 0; i < dev_arr_size; i++) {
    if (host_arr[i] < static_cast<T>(val)) {
      std::cout << "host_arr[" << i << "]:" << host_arr[i] << " val:" << val
                << "\n";
      return false;
    }
  }
  return true;
}


}

#endif
