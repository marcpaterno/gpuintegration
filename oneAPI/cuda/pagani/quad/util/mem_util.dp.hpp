#ifndef MEM_UTIL_CUH
#define MEM_UTIL_CUH

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

#include <dpct/dpl_utils.hpp>

template<typename T>
void
cuda_memcpy_to_host(T* dest, T* src, size_t size){
    /*
    DPCT1003:99: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    auto rc =
      (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
    if (rc != 0)
        throw std::bad_alloc();
}

template <typename T>
void
cuda_memcpy_to_device(T* dest, T* src, size_t size) try {
    /*
    DPCT1003:100: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    auto rc =
      (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
    /*
    DPCT1000:102: Error handling if-stmt was detected but could not be
     * rewritten.
    */
    if (rc != 0)
        /*
        DPCT1001:101: The statement could not be removed.
        */
        throw std::bad_alloc();
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename T>
void
cuda_memcpy_device_to_device(T* dest, T* src, size_t size) try {
    /*
    DPCT1003:103: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    auto rc =
      (dpct::get_default_queue().memcpy(dest, src, sizeof(T) * size).wait(), 0);
    /*
    DPCT1000:105: Error handling if-stmt was detected but could not be
     * rewritten.
    */
    if (rc != 0)
        /*
        DPCT1001:104: The statement could not be removed.
        */
        throw std::bad_alloc();
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template<typename T>
struct Range{
  Range() = default;
  Range(T l, T h):low(l),high(h){}
  T low = 0., high = 0.;  
};

 template<typename T>
 
 void
 device_print_array(T* arr, size_t size){
    for(size_t i=0; i < size; ++i)
       printf("arr[%lu]:%i\n", i, arr[i]);  //can't print arbitrary types from device, must fix to do std::cout from host
 }
 
 template<typename T>
 void
 print_device_array(T* arr, size_t size){
    dpct::get_default_queue().parallel_for(
      sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
          device_print_array<T>(arr, size);
      });
     dpct::get_current_device().queues_wait_and_throw();
 }


 template<class T>
  T*
  host_alloc(size_t size){
    T* temp = new T[size];;  
    if (temp == nullptr){
      throw std::bad_alloc();
    }
    return temp;
  }

  template <class T>
  T*
  cuda_malloc(size_t size) try {
    T* temp;
    /*
    DPCT1003:108: Migrated API does not return error code. (*, 0) is
     * inserted. You may need to rewrite this code.
    */
    auto rc = (temp = (T*)sycl::malloc_device(sizeof(T) * size,
                                              dpct::get_default_queue()),
               0);
    /*
    DPCT1000:107: Error handling if-stmt was detected but could not be
     * rewritten.
    */
    if (rc != 0) {
      printf("device side\n");
      /*
      DPCT1001:106: The statement could not be removed.
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

  //rename to  free_and_reallocate, delete copy_size, unecessary
  template<typename T>
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
  
  template<typename T>
  void
  set_array_to_value(T* array, size_t size, T val, sycl::nd_item<3> item_ct1){
        size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
        if(tid < size){
            array[tid] = val;
        }
  }
     
   template<typename T>
  void
  set_array_range_to_value(T* array, size_t first_to_change, size_t last_to_change, size_t  total_size, T val,
                           sycl::nd_item<3> item_ct1){
        size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
        if(tid >= first_to_change && tid <= last_to_change && tid < total_size){
            array[tid] = val;          
        }
  } 
    
 template<typename T>
  void set_device_array_range(T* arr, size_t first_to_change, size_t last_to_change, size_t  size, T val){
    size_t num_threads = 64;
    size_t num_blocks = size/num_threads +  ((size % num_threads) ? 1 : 0);
    /*
    DPCT1049:109: The workgroup size passed to the SYCL kernel may exceed
     * the limit. To get the device limit, query
     * info::device::max_work_group_size. Adjust the workgroup size if needed.

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
    
  template<typename T>
  void set_device_array(T* arr, size_t size, T val){
    size_t num_threads = 64;
    size_t num_blocks = size/num_threads +  ((size % num_threads) ? 1 : 0);
    /*
    DPCT1049:110: The workgroup size passed to the SYCL kernel may exceed
     * the limit. To get the device limit, query
     * info::device::max_work_group_size. Adjust the workgroup size if needed.

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

template<typename T, typename C = T>
bool
array_values_smaller_than_val(T* dev_arr, size_t dev_arr_size, C val){
    double* host_arr = host_alloc<double>(dev_arr_size);
    dpct::get_default_queue()
      .memcpy(host_arr, dev_arr, sizeof(double) * dev_arr_size)
      .wait();

    for(int i = 0; i < dev_arr_size; i++){
        if(host_arr[i] >= static_cast<T>(val))
            return false;
    }
    return true;
}

template<typename T, typename C = T>
bool
array_values_larger_than_val(T* dev_arr, size_t dev_arr_size, C val){
    double* host_arr = host_alloc<double>(dev_arr_size);
    dpct::get_default_queue()
      .memcpy(host_arr, dev_arr, sizeof(double) * dev_arr_size)
      .wait();

    for(int i = 0; i < dev_arr_size; i++){
        if(host_arr[i] < static_cast<T>(val)){
            std::cout<<"host_arr["<<i<<"]:"<<host_arr[i]<<" val:"<<val<<"\n";
            return false;
        }
    }
    return true;
}


#endif