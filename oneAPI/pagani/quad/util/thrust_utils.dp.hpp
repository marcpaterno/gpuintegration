#ifndef QUAD_THRUST_UTILS_CUH
#define QUAD_THRUST_UTILS_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

template <typename T1, typename T2>
double
dot_product(T1* arr1, T2* arr2, const size_t size) {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
    dpct::device_pointer<T1> wrapped_mask_1 = dpct::get_device_pointer(arr1);
    dpct::device_pointer<T2> wrapped_mask_2 = dpct::get_device_pointer(arr2);
    double res =
      dpct::inner_product(oneapi::dpl::execution::make_device_policy(q_ct1),
						  wrapped_mask_1,
						  wrapped_mask_1 + size,
                          wrapped_mask_2,
                          0.);
    return res;
}

template<typename T>
T
reduction(T* arr, size_t size){
    dpct::device_pointer<T> wrapped_ptr = dpct::get_device_pointer(arr);
    return std::reduce(
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      wrapped_ptr,
      wrapped_ptr + size);
}


#endif