#ifndef QUAD_THRUST_UTILS_CUH
#define QUAD_THRUST_UTILS_CUH

#include <CL/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/stats.hpp"
#include "dpct-exp/common/cuda/custom_functions.dp.hpp"
#include "dpct-exp/common/cuda/cudaMemoryUtil.h"




// https://www.apriorit.com/dev-blog/614-cpp-cuda-accelerate-algorithm-cpu-gpu

template <typename T1, typename T2, bool use_custom = false>
double
dot_product(T1* arr1, T2* arr2, const size_t size)
{
  //dpct::device_ext& dev_ct1 = dpct::get_current_device();
  //sycl::queue& q_ct1 = dev_ct1.default_queue();
  sycl::queue q_ct1;
  if constexpr (use_custom == false) {
	 T1* res = sycl::malloc_shared<T1>(1, q_ct1);
		auto est_ev =
		  oneapi::mkl::blas::column_major::dot(q_ct1, size, arr1, 1, arr2, 1, res);
		est_ev.wait();
		double result = res[0];
		sycl::free(res, q_ct1);
		return result;
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
	auto q = dpct::get_default_queue();
    int64_t* min = sycl::malloc_shared<int64_t>(1, q);
    int64_t* max = sycl::malloc_shared<int64_t>(1, q);
    const int stride = 1;

    sycl::event est_ev =
      oneapi::mkl::blas::column_major::iamax(q, size, arr, stride, max);

    sycl::event est_ev2 =
      oneapi::mkl::blas::column_major::iamin(q, size, arr, stride, min);

    est_ev.wait();
    est_ev2.wait();

    quad::cuda_memcpy_to_host<T>(&range.low, &arr[min[0]], 1);
    quad::cuda_memcpy_to_host<T>(&range.high, &arr[max[0]], 1);
    free(min, q);
    free(max, q);
    return range;
  }

  auto res = min_max<T>(arr, size);
  range.low = res.first;
  range.high = res.second;
  return range;
}

#endif