#ifndef ONE_API_QUAD_UTIL_CUDAMEMORY_UTIL_H
#define ONE_API_QUAD_UTIL_CUDAMEMORY_UTIL_H

#include <CL/sycl.hpp>
#include <stdio.h>

#include "oneapi/mkl.hpp"
#include "oneapi/mkl/stats.hpp"

namespace quad {
  template <typename T>
  void
  set_device_array_range(sycl::queue& q,
                         T* arr,
                         const size_t& lindex,
                         const size_t& rindex,
                         const T& val)
  {
    if (rindex < lindex) {
      printf("r < right index %lu smaller than left index %lu index in "
             "quad::set_device_array_range:",
             rindex,
             lindex);
      exit(1);
    }

    q.submit([&](auto& cgh) {
       q.parallel_for(sycl::range<1>(rindex - lindex + 1),
                      [=](sycl::id<1> i) { arr[i + lindex] = val; });
     })
      .wait();
  }

  template <typename T>
  void
  parallel_fill(sycl::queue& q, T* arr, size_t size, const T& val)
  {
    q.submit([&](auto& cgh) {
       q.parallel_for(sycl::range<1>(size),
                      [=](sycl::id<1> i) { arr[i] = val; });
     })
      .wait();
  }

  template <typename IntegT>
  IntegT*
  make_gpu_integrand(const IntegT& integrand)
  {
    IntegT* d_integrand =
      sycl::malloc_shared<IntegT>(d_integrand, sizeof(IntegT));
    memcpy(d_integrand, integrand, sizeof(IntegT));
    return d_integrand;
  }

  size_t
  total_device_mem(sycl::queue& q, size_t num_regions)
  {
    auto device = q.get_device();
    return device.get_info<sycl::info::device::max_mem_alloc_size>();
  }

  template <typename T>
  T
  dot_product(sycl::queue& q, T* list_a, T* list_b, size_t size)
  {
    T* res = sycl::malloc_shared<T>(1, q);
    auto est_ev =
      oneapi::mkl::blas::row_major::dot(q, size, list_a, 1, list_b, 1, res);
    est_ev.wait();
    double result = res[0];
    sycl::free(res, q);
    return result;
  }

  template <typename T>
  T
  reduction(sycl::queue& q, T* list, size_t size)
  {
    T* res = sycl::malloc_shared<T>(1, q);
    auto even_n = oneapi::mkl::blas::row_major::asum(q, size, list, 1, res);
    even_n.wait();
    T result = res[0];
    sycl::free(res, q);
    return result;
  }

  template <typename T>
  T*
  copy_to_shared(sycl::queue& q, T* data, size_t size)
  {
    T* tmp = sycl::malloc_shared<T>(size, q);
    memcpy(tmp, data, sizeof(T) * size);
    return tmp;
  }

  template <typename T>
  struct Range {
    Range() = default;

    Range(const Range<T>& src)
    {
      low = src.low;
      high = src.high;
    }

    Range(T l, T h) : low(l), high(h) {}
    T low = 0., high = 0.;
  };

  template <typename T, typename C = T>
  bool
  array_values_larger_than_val(T* dev_arr, size_t dev_arr_size, C val)
  {

    for (int i = 0; i < dev_arr_size; i++) {
      if (dev_arr[i] < static_cast<T>(val)) {
        return false;
      }
    }

    return true;
  }
}

#endif
