#ifndef QUAD_THRUST_UTILS_CUH
#define QUAD_THRUST_UTILS_CUH

#include <CL/sycl.hpp>

#include "oneapi/mkl.hpp"
#include "oneapi/mkl/stats.hpp"


template <typename T1, typename T2>
double
dot_product(T1* arr1, T2* arr2, const size_t size) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q = dev_ct1.default_queue();
    T1* res = sycl::malloc_shared<T1>(1, q);
    auto est_ev = oneapi::mkl::blas::row_major::dot(q, size, arr1, 1, arr2, 1 , res);
    est_ev.wait();
    double result = res[0];
    sycl::free(res, q);
    return result;
}

template<typename T>
T
reduction(T* arr, size_t size){
    T res = dpl::experimental::reduce_async(dpl::execution::dpcpp_default, arr, arr + size).get();
    return res;
}


#endif