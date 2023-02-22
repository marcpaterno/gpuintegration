#ifndef QUAD_THRUST_UTILS_CUH
#define QUAD_THRUST_UTILS_CUH

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include "dpct-exp/common/cuda/custom_functions.dp.hpp"
#include "dpct-exp/common/cuda/cudaMemoryUtil.h"
#include <numeric>

// https://www.apriorit.com/dev-blog/614-cpp-cuda-accelerate-algorithm-cpu-gpu

template <typename T1, typename T2, bool use_custom = false>
double
dot_product(T1* arr1, T2* arr2, const size_t size)
{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();

  if constexpr (use_custom == false) {
    dpct::device_pointer<T1> wrapped_mask_1 = dpct::get_device_pointer(arr1);
    dpct::device_pointer<T2> wrapped_mask_2 = dpct::get_device_pointer(arr2);
    double res =
      dpct::inner_product(oneapi::dpl::execution::make_device_policy(q_ct1),
                          oneapi::dpl::execution::make_device_policy(q_ct1),
                          wrapped_mask_2,
                          wrapped_mask_2 + size,
                          wrapped_mask_1,
                          0.);

    return res;
  }

  double res = custom_inner_product_atomics<T1, T2>(arr1, arr2, size);
  return res;
}

template <typename T, bool use_custom = false>
T
reduction(T* arr, size_t size)
{
  if constexpr (use_custom == false) {
    dpct::device_pointer<T> wrapped_ptr = dpct::get_device_pointer(arr);
    return std::reduce(
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      wrapped_ptr,
      wrapped_ptr + size);
  }

  return custom_reduce_atomics(arr, size);
}

template <typename T, bool use_custom = false>
void
exclusive_scan(T* arr, size_t size, T* out)
{
  if constexpr (use_custom == false) {
    dpct::device_pointer<T> d_ptr = dpct::get_device_pointer(arr);
    dpct::device_pointer<T> scan_ptr = dpct::get_device_pointer(out);
    std::exclusive_scan(
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      d_ptr,
      d_ptr + size,
      scan_ptr,
      0);
  } else {
    sum_scan_blelloch(out, arr, size);
  }
}

template <typename T>
void
thrust_exclusive_scan(T* arr, size_t size, T* out)
{
  dpct::device_pointer<T> d_ptr = dpct::get_device_pointer(arr);
  dpct::device_pointer<T> scan_ptr = dpct::get_device_pointer(out);
  std::exclusive_scan(
    oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
    d_ptr,
    d_ptr + size,
    scan_ptr,
    0);
}

template <typename T, bool use_custom = false>
quad::Range<T>
device_array_min_max(T* arr, size_t size)
{
  quad::Range<T> range;
  if (use_custom == false) {
    dpct::device_pointer<T> d_ptrE = dpct::get_device_pointer(arr);
    /*
    DPCT1007:149: Migration of this CUDA API is not supported by the Intel(R)
    DPC++ Compatibility Tool.
    */
    auto __tuple = thrust::minmax_element(d_ptrE, d_ptrE + size);
    range.low = *__tuple.first;
    range.high = *__tuple.second;
    return range;
  }

  auto res = min_max<T>(arr, size);
  range.low = res.first;
  range.high = res.second;
  return range;
}

#endif