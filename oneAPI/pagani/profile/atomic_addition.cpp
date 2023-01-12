#include <oneapi/dpl/execution>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>
#include <numeric>
 
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
T*
copy_to_host(T* dest, T* src, size_t size){
	//sycl::queue q_ct1(sycl::gpu_selector());
	sycl::queue q_ct1;
    q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
	return dest;
}

template<class T>
T*
cuda_malloc(size_t size){
	sycl::queue q_ct1;
    T* temp = sycl::malloc_device<T>(size, q_ct1);
    return temp;
}
  
template<typename T>
void
copy_to_device(T* dest, T* src, size_t size){
	sycl::queue q_ct1;
    q_ct1.memcpy(dest, src, sizeof(T) * size).wait();
}

template<typename T>
T*
alloc_and_copy_to_device(T* src, size_t size){
	T* tmp = cuda_malloc<T>(size);
	copy_to_device<T>(tmp, src, size);
	return tmp;
}

void
atomic_addition(double* src, double* out, size_t size, size_t num_blocks, size_t num_threads){
	sycl::queue q(sycl::gpu_selector(), sycl::property::queue::enable_profiling{});
		
    sycl::event e = q.submit([&](sycl::handler& cgh) {
		cgh.parallel_for(
              sycl::nd_range(sycl::range(num_blocks*num_threads) , sycl::range(num_threads)),
              [=](sycl::nd_item<1> item_ct1)[[intel::reqd_sub_group_size(32)]] {
				
				size_t tid = item_ct1.get_group(0) * num_threads + item_ct1.get_local_id(0);
				size_t total_num_threads = num_threads * num_blocks; 
	
				for(size_t i = tid; i < size; i += total_num_threads){
					
					for(int i=0; i < 8;++i){
					auto v = sycl::atomic_ref<double, 
						sycl::memory_order::relaxed, 
						sycl::memory_scope::device,
						sycl::access::address_space::global_space>(out[item_ct1.get_local_id(0)]);
					v += src[i];
					}
					//dpct::atomic_fetch_add(&out[item_ct1.get_local_id(0)], src[i]);
					//out[item_ct1.get_local_id(0)] += src[i];
				}
			
		});
    });
		
	q.wait();
    
	double time = (e.template get_profiling_info<sycl::info::event_profiling::command_end>()  -   
	e.template get_profiling_info<sycl::info::event_profiling::command_start>());
	std::cout<< "time:" << std::scientific << time/1.e6 << std::endl;
}

int main(){
    
	const size_t num_threads = 64;

	std::vector<double> src;
	src.resize(32768 * 1025 * 2);
	std::iota(src.begin(), src.end(), 1.);
	
	std::array<double, num_threads> output = {0.};
	
	std::cout<<"size:"<<src.size()<<std::endl;
	std::cout<<"Memory:"<<src.size()*8/1e9<<"GB\n";
	
	double* d_src = alloc_and_copy_to_device<double>(src.data(), src.size());
	double* d_output = alloc_and_copy_to_device<double>(output.data(), output.size());
	
	size_t num_blocks = src.size()/num_threads;
		
	atomic_addition(d_src, d_output, src.size(), num_blocks, num_threads);
		
	copy_to_host(output.data(), d_output, output.size());
	
	for(int i = 0; i < output.size(); ++i)
		printf("output %i, %e\n", i, output[i]);
	
	sycl::queue q_ct1;
	sycl::free(d_src, q_ct1);
	sycl::free(d_output, q_ct1);
	//ShowDevice(q_ct1);
    return 0;
}
