#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

void profile(double* block_results, size_t num_blocks){
	size_t block_size = 64;
  	sycl::queue q(sycl::gpu_selector(), sycl::property::queue::enable_profiling{});
    sycl::event e = q.submit([&](sycl::handler& cgh) {
				
        sycl::accessor<double, 1,
                        sycl::access_mode::read_write,
                        sycl::access::target::local>
            shared_acc(sycl::range(8), cgh); 
			
		cgh.parallel_for(
              sycl::nd_range(sycl::range(num_blocks*block_size) , sycl::range(block_size)),
              [=](sycl::nd_item<1> item_ct1)
                [[intel::reqd_sub_group_size(32)]] {
			
			const size_t tid = item_ct1.get_local_id(0) + item_ct1.get_group(0) * item_ct1.get_local_range().get(0);
			double val = static_cast<double>(tid);
			item_ct1.barrier();
			val = quad::blockReduceSum<double>(val, item_ct1, shared_acc.get_pointer());
			item_ct1.barrier();
			if(item_ct1.get_local_id(0)  == 0)
				block_results[item_ct1.get_group(0)] = val;	
		});
	});
	q.wait();
}	

double compute_expected(size_t num_blocks, size_t num_threads){
	size_t res = 0;
	for(int i=0; i < num_blocks * num_threads; ++i)
		res += i;
	return static_cast<double>(res);
}

int main(){
	
	size_t num_blocks = 262144*4;
	size_t num_threads = 64;
	double* block_res = cuda_malloc<double>(num_blocks);
	profile(block_res, num_blocks);
	double res = reduction<double>(block_res, num_blocks);
	printf("res:%e expected:%e\n", res, compute_expected(num_blocks, num_threads));
    return 0;
}

