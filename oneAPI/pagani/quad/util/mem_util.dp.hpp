#ifndef MEM_UTIL_CUH
#define MEM_UTIL_CUH

//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>

void ShowDevice(sycl::queue &q) {
      using namespace sycl;
      // Output platform and device information.
      auto device = q.get_device();
      auto p_name = device.get_platform().get_info<info::platform::name>();
      std::cout << "Platform Name: " << p_name << "\n";
      auto p_version = device.get_platform().get_info<info::platform::version>();
      std::cout << "Platform Version: " << p_version << "\n";
      auto d_name = device.get_info<info::device::name>();
      std::cout << "Device Name: " << d_name << "\n";
      auto max_work_group = device.get_info<info::device::max_work_group_size>();
        
      auto max_compute_units = device.get_info<info::device::max_compute_units>();
      std::cout << "Max Compute Units: " << max_compute_units << "\n\n";
      std::cout << "max_mem_alloc_size " << device.get_info<sycl::info::device::max_mem_alloc_size>() << std::endl;
      std::cout << "local_mem_size " <<  device.get_info<sycl::info::device::local_mem_size>() << std::endl;
}

template<typename T>
void
cuda_memcpy_to_host(T* dest, T* src, size_t size){
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
}

template <typename T>
void
cuda_memcpy_to_device(T* dest, T* src, size_t size){
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
}

template <typename T>
void
cuda_memcpy_device_to_device(T* dest, T* src, size_t size){
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
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
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  q_ct1.parallel_for(
      sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
          device_print_array<T>(arr, size);
      }).wait();
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
cuda_malloc(size_t size){
    T* temp;
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    temp = sycl::malloc_device<T>(size, q_ct1);
    return temp;
}

//candidate for deletion
template<typename T>
void
ExpandcuArray(T*& array, int currentSize, int newSize)
{
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    int copy_size = std::min(currentSize, newSize);
    T* temp = cuda_malloc<T>(newSize);
    sycl::free(array, q_ct1);
    array = temp;
}

template <typename IntegT>
IntegT*
make_gpu_integrand(const IntegT& integrand)
{
    IntegT* d_integrand;
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    d_integrand =
      (IntegT*)sycl::malloc_shared(sizeof(IntegT), q_ct1);
    //memcpy(d_integrand, &integrand, sizeof(IntegT));
	new (d_integrand) IntegT(integrand);
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
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    q_ct1.parallel_for(
      sycl::nd_range(sycl::range(1, 1, num_blocks) *
                       sycl::range(1, 1, num_threads),
                     sycl::range(1, 1, num_threads)),
      [=](sycl::nd_item<3> item_ct1) {
          set_array_range_to_value<T>(
            arr, first_to_change, last_to_change, size, val, item_ct1);
      }).wait();
}   
    
template<typename T>
void set_device_array(T* arr, size_t size, T val){
    size_t num_threads = 64;
    size_t num_blocks = size/num_threads +  ((size % num_threads) ? 1 : 0);
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    q_ct1.parallel_for(
      sycl::nd_range(sycl::range(1, 1, num_blocks) *
                       sycl::range(1, 1, num_threads),
                     sycl::range(1, 1, num_threads)),
      [=](sycl::nd_item<3> item_ct1) {
          set_array_to_value<T>(arr, size, val, item_ct1);
      }).wait();
}

template<typename T, typename C = T>
bool
array_values_smaller_than_val(T* dev_arr, size_t dev_arr_size, C val){
    double* host_arr = host_alloc<double>(dev_arr_size);
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    q_ct1.memcpy(host_arr, dev_arr, sizeof(double) * dev_arr_size)
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
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());
    q_ct1.memcpy(host_arr, dev_arr, sizeof(double) * dev_arr_size)
      .wait();

    for(int i = 0; i < dev_arr_size; i++){
        if(host_arr[i] < static_cast<T>(val)){
            return false;
        }
    }
    return true;
}


#endif
