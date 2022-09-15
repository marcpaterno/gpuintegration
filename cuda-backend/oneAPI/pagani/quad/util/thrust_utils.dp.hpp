#ifndef QUAD_THRUST_UTILS_CUH
#define QUAD_THRUST_UTILS_CUH

#include <CL/sycl.hpp>

#include "oneapi/mkl.hpp"
#include "oneapi/mkl/stats.hpp"
#include "oneAPI/pagani/quad/util/custom_functions.dp.hpp"


template <typename T1, typename T2, bool use_custom = false>
double
dot_product(T1* arr1, T2* arr2, const size_t size) {
	
	
	sycl::gpu_selector gpuSelector;
	sycl::queue q(gpuSelector);
	
    //dpct::device_ext& dev_ct1 = dpct::get_current_device();
    //sycl::queue& q = dev_ct1.default_queue();
    if constexpr(use_custom == false ){
        T1* res = sycl::malloc_shared<T1>(1, q);
        auto est_ev = oneapi::mkl::blas::column_major::dot(q, size, arr1, 1, arr2, 1 , res);
        est_ev.wait();
        double result = res[0];
        sycl::free(res, q);
        return result;
    }
    double res = custom_inner_product_atomics<T1, T2>(arr1, arr2, size);
	return res;
}

template<typename T, bool use_custom = false>
T
reduction(T* arr, size_t size){
    if constexpr(use_custom == false ){
        T res = dpl::experimental::reduce_async(dpl::execution::dpcpp_default, arr, arr + size).get();
        return res;
    }
    return custom_reduce_atomics(arr, size);
}

template<typename T, bool use_custom = false>
void
exclusive_scan(T* arr, size_t size, T* out){
	if constexpr(use_custom == false ){
                /*dpct::device_pointer<T> d_ptr = dpct::get_device_pointer(arr);
                dpct::device_pointer<T> scan_ptr = dpct::get_device_pointer(out);
                std::exclusive_scan(oneapi::dpl::execution::make_device_policy(
                                      dpct::get_default_queue()),
                                    d_ptr,
                                    d_ptr + size,
                                    scan_ptr,
                                    0);*/
                dpct::device_ext& dev_ct1 = dpct::get_current_device();
                sycl::queue& q_ct1 = dev_ct1.default_queue();                    
                dpl::experimental::exclusive_scan_async(oneapi::dpl::execution::make_device_policy(q_ct1), arr, arr + size, out, 0.).wait();
        }
	else{
		sum_scan_blelloch(out, arr, size);
	}
}

template<typename T>
void
thrust_exclusive_scan(T* arr, size_t size, T* out){
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
    dpl::experimental::exclusive_scan_async(oneapi::dpl::execution::make_device_policy(q_ct1), arr, arr + size, out, 0.).wait();
}

/*template<typename T, bool use_custom = false>
Range<T>
device_array_min_max(T* arr, size_t size){
    Range<T> range;
    if constexpr(use_custom == false){
        auto q = dpct::get_default_queue();
        double* min = sycl::malloc_shared<double>(1, q);  
        double* max = sycl::malloc_shared<double>(1, q);    

        oneapi::mkl::stats::dataset<oneapi::mkl::stats::layout::column_major, T*> wrapper(1, size, arr);

        auto this_event = oneapi::mkl::stats::min_max<oneapi::mkl::stats::method::fast, 
                            double, oneapi::mkl::stats::layout::column_major>(q, wrapper, min, max);
        this_event.wait();

        range.low = min[0];
        range.high = max[0];
        free(min, q);
        free(max, q);
        return range;
    }
    auto res = min_max<T>(arr, size);
	range.low  = res.first;
	range.high = res.second;
	return range;
}*/

template<typename T, bool use_custom = false, bool cuda_backend = true>
Range<T>
device_array_min_max(T* arr, size_t size){
    Range<T> range;
	if constexpr(use_custom == true && cuda_backend == true){
		auto q = dpct::get_default_queue();
		int64_t* min = sycl::malloc_shared<int64_t>(1, q);  
		int64_t* max = sycl::malloc_shared<int64_t>(1, q);    
		const int stride = 1;
		
		sycl::event est_ev = oneapi::mkl::blas::column_major::iamax(
			q, size, arr, stride, max);
						  
		sycl::event est_ev2 = oneapi::mkl::blas::column_major::iamin(
			q, size, arr, stride, min);
		
		est_ev.wait();
		est_ev2.wait();
		
		cuda_memcpy_to_host<T>(&range.low, &arr[min[0]], 1);
		cuda_memcpy_to_host<T>(&range.high, &arr[max[0]], 1);
		free(min, q);
		free(max, q);
		return range;
	}
	
	if constexpr(use_custom == false && cuda_backend == false){
        auto q = dpct::get_default_queue();
        double* min = sycl::malloc_shared<double>(1, q);  
        double* max = sycl::malloc_shared<double>(1, q);    

        oneapi::mkl::stats::dataset<oneapi::mkl::stats::layout::row_major, T*> wrapper(1, size, arr);

        auto this_event = oneapi::mkl::stats::min_max<oneapi::mkl::stats::method::fast, 
                            double, oneapi::mkl::stats::layout::row_major>(q, wrapper, min, max);
        this_event.wait();

        range.low = min[0];
        range.high = max[0];
        free(min, q);
        free(max, q);
        return range;
    }
	
	
	
	auto res = min_max<T>(arr, size);
	range.low  = res.first;
	range.high = res.second;
	return range;
}


#endif