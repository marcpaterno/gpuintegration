#ifndef ONE_API_SUB_REGION_FILTER_H
#define ONE_API_SUB_REGION_FILTER_H

//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/Region_characteristics.h"



template<size_t ndim>
class Sub_regions_filter{
  using Estimates = Region_estimates<ndim>;
  using Sub_regs = Sub_regions<ndim>;
  using Reg_chars = Region_characteristics<ndim>;


  public:
    ~Sub_regions_filter();
    Sub_regions_filter(sycl::queue& q, const size_t num_regions);
    
    size_t 
    compute_num_blocks(const size_t num_regions)const;
    
    size_t filter(sycl::queue& q, Sub_regions<ndim>& sub_regions, 
              Reg_chars& region_characteristics, 
              const Estimates& region_ests, 
              Estimates& parent_ests);
    void
    alignRegions(sycl::queue& q, 
                 double* dRegions,
                 double* dRegionsLength,
                 double* activeRegions,
                 double* dRegionsIntegral,
                 double* dRegionsError,
                 double* dRegionsParentIntegral,
                 double* dRegionsParentError,
                 int* subDividingDimension,
                 double* scannedArray,
                 double* newActiveRegions,
                 double* newActiveRegionsLength,
                 int* newActiveRegionsBisectDim,
                 size_t numRegions,
                 size_t newNumRegions,
                 int numOfDivisionOnDimension);

    size_t
    get_num_active_regions(sycl::queue& q, Region_characteristics<ndim>& regs);
    
    double* scanned_array = nullptr;
    sycl::queue* _q;
};

//decouple num_active_regions and setting the scanned_array

template<size_t ndim>
size_t Sub_regions_filter<ndim>::get_num_active_regions(sycl::queue& q, Region_characteristics<ndim>& regs){

    const size_t num_regions = regs.size;
    double* active_regions = regs.active_regions;
    dpl::experimental::exclusive_scan_async(oneapi::dpl::execution::make_device_policy(q),
        active_regions, active_regions + num_regions, scanned_array, 0.).wait();
    size_t num_active = scanned_array[num_regions-1];
	
	double last;
	quad::copy_to_host<double>(&last, &active_regions[num_regions-1], 1);
    if (last == 1)
        num_active++;
    return num_active;
}

template<size_t ndim>
size_t
Sub_regions_filter<ndim>::compute_num_blocks(const size_t num_regions)const{
    size_t numThreads = 64;
    return num_regions / numThreads + ((num_regions % numThreads) ? 1 : 0);
}

template<size_t ndim>
Sub_regions_filter<ndim>::Sub_regions_filter(sycl::queue& q, const size_t num_regions){
    scanned_array = sycl::malloc_shared<double>(num_regions, q);
    _q = &q;
}

template<size_t ndim>
Sub_regions_filter<ndim>::~Sub_regions_filter(){
    sycl::free(scanned_array, *_q);
}

template<size_t ndim>
void
Sub_regions_filter<ndim>::alignRegions(sycl::queue& q, 
             double* dRegions,
             double* dRegionsLength,
             double* activeRegions,
             double* dRegionsIntegral,
             double* dRegionsError,
             double* dRegionsParentIntegral,
             double* dRegionsParentError,
             int* subDividingDimension,
             double* scannedArray,
             double* newActiveRegions,
             double* newActiveRegionsLength,
             int* newActiveRegionsBisectDim,
             size_t numRegions,
             size_t newNumRegions,
             int numOfDivisionOnDimension)
{
      q.submit([&](auto &cgh) {
    
        //sycl::stream str(262144, 4096, cgh);
        cgh.parallel_for(sycl::range<1>(numRegions), [=](sycl::id<1> tid){
            if (activeRegions[tid[0]] == 1.) {
                size_t interval_index = static_cast<size_t>(scannedArray[tid[0]]);

                for (size_t i = 0; i < ndim; ++i) {
                    newActiveRegions[i * newNumRegions + interval_index] = dRegions[i * numRegions + tid[0]];
                    newActiveRegionsLength[i * newNumRegions + interval_index] = dRegionsLength[i * numRegions + tid[0]];
              }

              dRegionsParentIntegral[interval_index] = dRegionsIntegral[tid[0]];
              dRegionsParentError[interval_index] = dRegionsError[tid[0]];

              for (size_t i = 0; i < numOfDivisionOnDimension; ++i) {
                newActiveRegionsBisectDim[i * newNumRegions + interval_index] = subDividingDimension[tid[0]];
              }
            }
        });
    }).wait();
}

template<size_t ndim>
size_t
Sub_regions_filter<ndim>::filter(sycl::queue& q, 
              Sub_regs& sub_regions, 
              Reg_chars& region_characteristics, 
              const Estimates& region_ests, 
              Estimates& parent_ests){
  
        const size_t current_num_regions = sub_regions.size;
        const size_t num_active_regions = get_num_active_regions(q, region_characteristics);

        if(num_active_regions == 0){
            return 0;
        }

        double* filtered_leftCoord = sycl::malloc_shared<double>(num_active_regions*ndim, q);
        double* filtered_length = sycl::malloc_shared<double>(num_active_regions*ndim, q);
        int* filtered_sub_dividing_dim = sycl::malloc_shared<int>(num_active_regions, q);

        parent_ests.reallocate(q, num_active_regions);
        const int numOfDivisionOnDimension = 1;
        const size_t num_blocks = compute_num_blocks(current_num_regions);
        alignRegions(q, sub_regions.dLeftCoord,
                                      sub_regions.dLength,
                                      region_characteristics.active_regions,
                                      region_ests.integral_estimates,
                                      region_ests.error_estimates,
                                      parent_ests.integral_estimates,
                                      parent_ests.error_estimates,
                                      region_characteristics.sub_dividing_dim,
                                      scanned_array,
                                      filtered_leftCoord,
                                      filtered_length,
                                      filtered_sub_dividing_dim,
                                      current_num_regions,
                                      num_active_regions,
                                      numOfDivisionOnDimension);
        sycl::free(sub_regions.dLeftCoord, q);
        sycl::free(sub_regions.dLength, q);
        sycl::free(region_characteristics.sub_dividing_dim, q);
        
        sub_regions.dLeftCoord = filtered_leftCoord;
        sub_regions.dLength = filtered_length;
        region_characteristics.sub_dividing_dim = filtered_sub_dividing_dim;
        sub_regions.size = num_active_regions;
        region_characteristics.size = num_active_regions;
        return num_active_regions;
    }
#endif
