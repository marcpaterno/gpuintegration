#ifndef SUB_REGION_FILTER_CUH
#define SUB_REGION_FILTER_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/pagani/quad/GPUquad/Sub_regions.dp.hpp"
#include "oneAPI/pagani/quad/util/mem_util.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/heuristic_classifier.dp.hpp"
#include <numeric>

template <typename T, int NDIM>
void
alignRegions(T* dRegions,
               T* dRegionsLength,
               double* activeRegions,
               T* dRegionsIntegral,
               T* dRegionsError,
               T* dRegionsParentIntegral,
               T* dRegionsParentError,
               int* subDividingDimension,
               double* scannedArray,
               T* newActiveRegions,
               T* newActiveRegionsLength,
               int* newActiveRegionsBisectDim,
               size_t numRegions,
               size_t newNumRegions,
               int numOfDivisionOnDimension,
               sycl::nd_item<3> item_ct1)
{

    size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);

    if (tid < numRegions && activeRegions[tid] == 1.) {
      size_t interval_index = scannedArray[tid];

      for (int i = 0; i < NDIM; ++i) {
        newActiveRegions[i * newNumRegions + interval_index] =
          dRegions[i * numRegions + tid];
        newActiveRegionsLength[i * newNumRegions + interval_index] =
          dRegionsLength[i * numRegions + tid];
      }

      dRegionsParentIntegral[interval_index] =
        dRegionsIntegral[tid];
      dRegionsParentError[interval_index] = dRegionsError[tid];

      for (int i = 0; i < numOfDivisionOnDimension; ++i) {
        newActiveRegionsBisectDim[i * newNumRegions + interval_index] =
          subDividingDimension[tid];
      }
    }
}


template<size_t ndim, bool use_custom = false>
class Sub_regions_filter{
    public:
    
    using Regions = Sub_regions<ndim>;
    using Region_char = Region_characteristics<ndim>;
    using Region_ests = Region_estimates<ndim>;
    
    Sub_regions_filter(const size_t num_regions){
        scanned_array = cuda_malloc<double>(num_regions);
    }
    
    
    
    size_t
    get_num_active_regions(double* active_regions, const size_t num_regions) {
        dpct::device_ext& dev_ct1 = dpct::get_current_device();
        sycl::queue& q_ct1 = dev_ct1.default_queue();
        
        /* dpct::device_pointer<double> d_ptr = dpct::get_device_pointer(active_regions);
           dpct::device_pointer<double> scan_ptr =
           dpct::get_device_pointer(scanned_array);
           std::exclusive_scan(oneapi::dpl::execution::make_device_policy(q_ct1),
                            d_ptr,
                            d_ptr + num_regions,
                            scan_ptr,
                            0);*/
                                    
        exclusive_scan<double, use_custom>(active_regions, num_regions, scanned_array);
        //dpl::experimental::exclusive_scan_async(oneapi::dpl::execution::make_device_policy(q_ct1), active_regions, active_regions + num_regions, scanned_array, 0.).wait();
        //size_t num_active = scanned_array[num_regions-1];
        
        double last_element;
        double num_active = 0;
        
        q_ct1.memcpy(&last_element, active_regions + num_regions - 1, sizeof(double));

        q_ct1.memcpy(&num_active, scanned_array + num_regions - 1, sizeof(double))
          .wait();
        
        
        if (last_element == 1.)
            num_active += 1;
        return static_cast<size_t>(num_active);
    }
    
    //filter out finished regions
    size_t
    filter(Regions* sub_regions,
           Region_char* region_characteristics,
           const Region_ests* region_ests,
           Region_ests* parent_ests) {
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();

        const size_t current_num_regions = sub_regions->size;
        const size_t num_active_regions = get_num_active_regions(region_characteristics->active_regions, current_num_regions);

        if(num_active_regions == 0){
            return 0;
        }
                
        //I dont' create Regions filtered_regions, because upon destruction it would deallocate and for performance reasons, I don't want a deep_copy to occur here
        double* filtered_leftCoord = cuda_malloc<double>(num_active_regions*ndim);        
        double* filtered_length = cuda_malloc<double>(num_active_regions*ndim);
        int* filtered_sub_dividing_dim = cuda_malloc<int>(num_active_regions);    
        
        parent_ests->reallocate(num_active_regions);
        const int numOfDivisionOnDimension = 1;
        const size_t num_blocks = compute_num_blocks(current_num_regions);
        
        auto dLeftCoord = sub_regions->dLeftCoord;
        auto dLength = sub_regions->dLength;
        auto active_regions = region_characteristics->active_regions;
        auto integral_estimates = region_ests->integral_estimates;
        auto error_estimates = region_ests->error_estimates;
        auto parent_integral_ests = parent_ests->integral_estimates;
        auto parent_error_ests = parent_ests->error_estimates;
        auto sub_dividing_dim = region_characteristics->sub_dividing_dim;
        
        q_ct1.submit([&](sycl::handler& cgh) {
            auto scanned_array_ct8 = scanned_array;

            cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, num_blocks) *
                                              sycl::range(1, 1, BLOCK_SIZE),
                                            sycl::range(1, 1, BLOCK_SIZE)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 alignRegions<double, static_cast<int>(ndim)>(
                                   dLeftCoord,
                                   dLength,
                                   active_regions,
                                   integral_estimates,
                                   error_estimates,
                                   parent_integral_ests,
                                   parent_error_ests,
                                   sub_dividing_dim,
                                   scanned_array_ct8,
                                   filtered_leftCoord,
                                   filtered_length,
                                   filtered_sub_dividing_dim,
                                   current_num_regions,
                                   num_active_regions,
                                   numOfDivisionOnDimension,
                                   item_ct1);
                             });
        });

        dev_ct1.queues_wait_and_throw();
        sycl::free(sub_regions->dLeftCoord, q_ct1);
        sycl::free(sub_regions->dLength, q_ct1);
        sycl::free(region_characteristics->sub_dividing_dim, q_ct1);
        sub_regions->dLeftCoord = filtered_leftCoord;
        sub_regions->dLength = filtered_length;
        region_characteristics->sub_dividing_dim = filtered_sub_dividing_dim;
        sub_regions->size = num_active_regions;
        region_characteristics->size = num_active_regions;
        quad::CudaCheckError();   
        return num_active_regions;
    }
    
    size_t compute_num_blocks(const size_t num_regions)const{
        size_t numThreads = BLOCK_SIZE;
        return num_regions / numThreads + ((num_regions % numThreads) ? 1 : 0);
    }
    
    ~Sub_regions_filter(){
        sycl::free(scanned_array, dpct::get_default_queue());
    }
    
    double* scanned_array = nullptr;
};

#endif