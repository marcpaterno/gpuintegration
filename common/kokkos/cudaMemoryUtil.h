#ifndef KOKKOS_PAGANI_KOKKOS_QUAD_UTIL_CUDAMEMORY_UTIL_H
#define KOKKOS_PAGANI_KOKKOS_QUAD_UTIL_CUDAMEMORY_UTIL_H
#include <Kokkos_Core.hpp>
#include <fstream>

typedef Kokkos::View<int*, Kokkos::CudaSpace> ViewVectorInt;
typedef Kokkos::View<float*, Kokkos::CudaSpace> ViewVectorFloat;
typedef Kokkos::View<double*, Kokkos::CudaSpace> ViewVectorDouble;
typedef Kokkos::
  View<double*, Kokkos::CudaSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    ViewVectorDoubleNoMang;
typedef Kokkos::View<size_t*, Kokkos::CudaSpace> ViewVectorSize_t;
//-------------------------------------------------------------------------------
// Const Device views
typedef Kokkos::View<const double*, Kokkos::CudaSpace> constViewVectorDouble;
typedef Kokkos::View<const int*, Kokkos::CudaSpace> constViewVectorInt;
typedef Kokkos::View<const size_t*, Kokkos::CudaSpace> constViewVectorSize_t;
//-------------------------------------------------------------------------------
// policies
typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;
//-------------------------------------------------------------------------------
// Shared Memory
typedef Kokkos::View<double*,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  ScratchViewDouble;
typedef Kokkos::View<int*,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  ScratchViewInt;

//-------------------------------------------------------------------------------
// Host views
typedef Kokkos::View<int*, Kokkos::Serial> HostVectorInt;
typedef Kokkos::View<double*, Kokkos::Serial> HostVectorDouble;
typedef Kokkos::View<size_t*, Kokkos::Serial> HostVectorSize_t;
//-------------------------------------------------------------------------------
typedef Kokkos::View<double*, Kokkos::CudaUVMSpace> ViewDouble;

template <int debug = 0>
class Recorder {
public:
  std::ofstream outfile;

  Recorder() = default;

  Recorder(std::string filename)
  {
    if constexpr (debug > 0)
      outfile.open(filename.c_str());
  }

  ~Recorder()
  {
    if constexpr (debug > 0)
      outfile.close();
  }
};

namespace quad {

  size_t
  GetAmountFreeMem()
  {
    size_t free_physmem, total_physmem;
    cudaMemGetInfo(&free_physmem, &total_physmem);
    return free_physmem;
  }

  template <class T>
  T*
  cuda_malloc_managed(size_t size)
  {
    Kokkos::View<T*, Kokkos::CudaUVMSpace> temp("temp", size);
    return temp;
  }

  template <class T>
  T*
  cuda_malloc_managed()
  {
    T* temp = nullptr;
    auto rc = cudaMallocManaged(&temp, sizeof(T));
    if (rc != cudaSuccess) {
      size_t free_physmem, total_physmem;
      cudaMemGetInfo(&free_physmem, &total_physmem);
      printf("cuda_malloc_managed() allocating size %lu free mem:%lu\n",
             sizeof(T),
             free_physmem);
      throw std::bad_alloc();
    }

    return temp;
  }

  template <typename T>
  T*
  cuda_copy_to_managed(T const& on_host)
  {
    T* buffer = (T*)(Kokkos::kokkos_malloc<Kokkos::CudaUVMSpace>(sizeof(T)));
    try {
      new (buffer) T(on_host);
    }
    catch (...) {
      Kokkos::kokkos_free(buffer);
      throw;
    }
    return buffer;
  }

  template <class T>
  Kokkos::View<T*, Kokkos::CudaSpace>
  cuda_malloc(size_t size)
  {
    Kokkos::View<T*, Kokkos::CudaSpace> temp("temp", size);
    return temp;
  }

  template <typename T>
  void
  cuda_memcpy_to_device(T* dest, T* src, size_t size)
  {
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
      printf("error in cuda_mempcy_to_device with host src\n");
      throw std::bad_alloc();
      abort();
    }
  }

  template <typename T>
  void
  cuda_memcpy_to_device(T* dest, const T* src, size_t size)
  {
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
      printf("error in cuda_mempcy_to_device with host src\n");
      throw std::bad_alloc();
      abort();
    }
  }

  template <typename T>
  void
  cuda_memcpy_device_to_device(T* dest, T* src, size_t size)
  {
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToDevice);
    if (rc != cudaSuccess) {
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
    return buffer;
  }

  size_t
  get_free_mem()
  {
    size_t free_physmem, total_physmem;
    cudaMemGetInfo(&free_physmem, &total_physmem);
    return free_physmem;
  }

  template <typename T>
  T*
  copy_to_host(T* src, size_t size)
  {
    T* dest = new T[size];
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess)
      throw std::bad_alloc();
    return dest;
  }

  template <typename T>
  void
  cuda_memcpy_to_host(T* dest, T const* src, size_t n_elements)
  {
    auto rc =
      cudaMemcpy(dest, src, sizeof(T) * n_elements, cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess)
      throw std::bad_alloc();
  }

  template <typename T>
  void
  cuda_memcpy_device_to_device(T* dest, T const* src, size_t size)
  {
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToDevice);
    if (rc != cudaSuccess)
      throw std::bad_alloc();
  }

  template <typename T>
  struct Range {
    Range() = default;
    Range(T l, T h) : low(l), high(h) {}
    T low = 0., high = 0.;
  };

  template <typename T>
  __global__ void
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
    device_print_array<T><<<1, 1>>>(arr, size);
    cudaDeviceSynchronize();
  }

  template <class T>
  T*
  host_alloc(size_t size)
  {
    T* temp = new T[size];
    if (temp == nullptr) {
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
    cudaFree(array);
    array = temp;
  }

  template <typename IntegT>
  IntegT*
  make_gpu_integrand(const IntegT& integrand)
  {
    return cuda_copy_to_managed<IntegT>(integrand);
  }

  template <typename T>
  void
  set_array_to_value(T* array, size_t size, T val)
  {
    Kokkos::parallel_for(
      "Loop1", size, KOKKOS_LAMBDA(const int& i) { array[i] = val; });
  }

  template <typename T>
  void
  set_array_range_to_value(T* array,
                           size_t first_to_change,
                           size_t last_to_change,
                           size_t size,
                           T val)
  {
    Kokkos::parallel_for(
      "Loop1", size, KOKKOS_LAMBDA(const int& i) {
        if (i >= first_to_change && i <= last_to_change) {
          array[i] = val;
        }
      });
  }

  template <typename T>
  void
  set_device_array(T* arr, size_t size, T val)
  {
    Kokkos::parallel_for(
      "Loop1", size, KOKKOS_LAMBDA(const int& i) { arr[i] = val; });
  }

  template <typename T, typename C = T>
  bool
  array_values_smaller_than_val(T* dev_arr, size_t size, C val)
  {
    size_t res = 0;
    Kokkos::parallel_reduce(
      "ProParRed1",
      size,
      KOKKOS_LAMBDA(const int64_t index, T& res) {
        if (dev_arr[index] > val)
          res += 1;
      },
      res);

    if (res == 0)
      return true;
    return false;
  }

  template <typename T, typename C = T>
  bool
  array_values_larger_than_val(T* dev_arr, size_t size, C val)
  {
    size_t res = 0;
    Kokkos::parallel_reduce(
      "ProParRed1",
      size,
      KOKKOS_LAMBDA(const int64_t index, T& res) {
        if (dev_arr[index] < val)
          res += 1;
      },
      res);

    if (res == 0)
      return true;
    return false;
  }

}

#endif
