#ifndef HYBRID_CUH
#define HYBRID_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

//#include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "pagani/quad/GPUquad/Region_characteristics.dp.hpp"
#include "pagani/quad/GPUquad/Region_estimates.dp.hpp"
#include "pagani/quad/GPUquad/Phases.dp.hpp"

template <size_t ndim>
void
two_level_errorest_and_relerr_classify(
  Region_estimates<ndim>* current_iter_raw_estimates,
  const Region_estimates<ndim>* prev_iter_two_level_estimates,
  const Region_characteristics<ndim>* reg_classifiers,
  double epsrel,
  bool relerr_classification = true) {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();

    size_t num_regions =  current_iter_raw_estimates->size;       
    //double epsrel = 1.e-3/*, epsabs = 1.e-12*/;    
    size_t block_size = 64;
    size_t numBlocks = num_regions / block_size + ((num_regions % block_size) ? 1 : 0);
    bool forbid_relerr_classification = !relerr_classification;
    if(prev_iter_two_level_estimates->size == 0){    
        //printf("B returning \n");
        return;
    }
        
    double* new_two_level_errorestimates = cuda_malloc<double>(num_regions);
    /*
    DPCT1049:111: The workgroup size passed to the SYCL kernel may exceed
     * the limit. To get the device limit, query
     * info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    q_ct1.parallel_for(sycl::nd_range(sycl::range(1, 1, numBlocks) *
                                        sycl::range(1, 1, block_size),
                                      sycl::range(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                           quad::RefineError<double>(
                             current_iter_raw_estimates->integral_estimates,
                             current_iter_raw_estimates->error_estimates,
                             prev_iter_two_level_estimates->integral_estimates,
                             prev_iter_two_level_estimates->error_estimates,
                             new_two_level_errorestimates,
                             reg_classifiers->active_regions,
                             num_regions,
                             epsrel,
                             forbid_relerr_classification,
                             item_ct1);
                       });
	
    dev_ct1.queues_wait_and_throw();
	//-----------------------------------
	//double* temp = quad::copy_to_host(reg_classifiers->active_regions, current_iter_raw_estimates->size);
	//for(int i=0; i < current_iter_raw_estimates->size; ++i)
	//	std::cout<<"region "<<i << " active:"<<temp[i]<<std::endl;
	
	//-----------------------------------
    sycl::free(current_iter_raw_estimates->error_estimates, q_ct1);
    current_iter_raw_estimates->error_estimates = new_two_level_errorestimates;
}

template <size_t ndim>
void
computute_two_level_errorest(
  Region_estimates<ndim>& current_iter_raw_estimates,
  const Region_estimates<ndim>& prev_iter_two_level_estimates,
  Region_characteristics<ndim>& reg_classifiers,
  bool relerr_classification = true) {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();

    size_t num_regions =  current_iter_raw_estimates.size;       
    double epsrel = 1.e-3/*, epsabs = 1.e-12*/;    
    size_t block_size = 64;
    size_t numBlocks = num_regions / block_size + ((num_regions % block_size) ? 1 : 0);
    bool forbid_relerr_classification = !relerr_classification;
    if(prev_iter_two_level_estimates.size == 0){    
        //printf("A returning \n");
        return;
    }
        
    double* new_two_level_errorestimates = cuda_malloc<double>(num_regions);
    /*
    DPCT1049:112: The workgroup size passed to the SYCL kernel may exceed
     * the limit. To get the device limit, query
     * info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    q_ct1.parallel_for(sycl::nd_range(sycl::range(1, 1, numBlocks) *
                                        sycl::range(1, 1, block_size),
                                      sycl::range(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                           quad::RefineError<double>(
                             current_iter_raw_estimates.integral_estimates,
                             current_iter_raw_estimates.error_estimates,
                             prev_iter_two_level_estimates.integral_estimates,
                             prev_iter_two_level_estimates.error_estimates,
                             new_two_level_errorestimates,
                             reg_classifiers.active_regions,
                             num_regions,
                             epsrel,
                             forbid_relerr_classification,
                             item_ct1);
                       });

    dev_ct1.queues_wait_and_throw();
    sycl::free(current_iter_raw_estimates.error_estimates, q_ct1);
    current_iter_raw_estimates.error_estimates = new_two_level_errorestimates;
}

  
#endif