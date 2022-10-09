#ifndef MEM_UTIL_CUH
#define MEM_UTIL_CUH

#include <iostream>
#include <new>
#include <cuda.h>

#if 0
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#endif

template <typename T>
void
cuda_memcpy_to_host(T* dest, T const* src, size_t n_elements)
{
  auto rc = cudaMemcpy(dest, src, sizeof(T) * n_elements, cudaMemcpyDeviceToHost);
  if (rc != cudaSuccess)
    throw std::bad_alloc();
}

template <typename T>
void
cuda_memcpy_to_device(T* dest, T const* src, size_t size)
{
  auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
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

template <class T>
T*
cuda_malloc(size_t size)
{
  T* temp;
  auto rc = cudaMalloc((void**)&temp, sizeof(T) * size);
  if (rc != cudaSuccess) {
    printf("device side\n");
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
  IntegT* d_integrand;
  cudaMallocManaged((void**)&d_integrand, sizeof(IntegT));
  memcpy(d_integrand, &integrand, sizeof(IntegT));
  return d_integrand;
}

template <typename T>
__global__ void
set_array_to_value(T* array, size_t size, T val)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    array[tid] = val;
  }
}

template <typename T>
__global__ void
set_array_range_to_value(T* array,
                         size_t first_to_change,
                         size_t last_to_change,
                         size_t total_size,
                         T val)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
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
  set_array_range_to_value<T><<<num_blocks, num_threads>>>(
    arr, first_to_change, last_to_change, size, val);
  cudaDeviceSynchronize();
}

template <typename T>
void
set_device_array(T* arr, size_t size, T val)
{
  size_t num_threads = 64;
  size_t num_blocks = size / num_threads + ((size % num_threads) ? 1 : 0);
  set_array_to_value<T><<<num_blocks, num_threads>>>(arr, size, val);
  cudaDeviceSynchronize();
}

template <typename T, typename C = T>
bool
array_values_smaller_than_val(T* dev_arr, size_t dev_arr_size, C val)
{
  double* host_arr = host_alloc<double>(dev_arr_size);
  cudaMemcpy(
    host_arr, dev_arr, sizeof(double) * dev_arr_size, cudaMemcpyDeviceToHost);

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
  cudaMemcpy(
    host_arr, dev_arr, sizeof(double) * dev_arr_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < dev_arr_size; i++) {
    if (host_arr[i] < static_cast<T>(val)) {
      std::cout << "host_arr[" << i << "]:" << host_arr[i] << " val:" << val
                << "\n";
      return false;
    }
  }
  return true;
}

#endif