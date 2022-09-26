#ifndef SUB_REGION_FILTER_CUH
#define SUB_REGION_FILTER_CUH

#include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "cuda/pagani/quad/util/mem_util.cuh"
#include "cuda/pagani/quad/GPUquad/heuristic_classifier.cuh"

template <typename T, int NDIM>
__global__ void
alignRegions(T* dRegions,
             T* dRegionsLength,
             int* activeRegions,
             T* dRegionsIntegral,
             T* dRegionsError,
             T* dRegionsParentIntegral,
             T* dRegionsParentError,
             int* subDividingDimension,
             int* scannedArray,
             T* newActiveRegions,
             T* newActiveRegionsLength,
             int* newActiveRegionsBisectDim,
             size_t numRegions,
             size_t newNumRegions,
             int numOfDivisionOnDimension)
{

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < numRegions && activeRegions[tid] == 1) {
    size_t interval_index = scannedArray[tid];

    for (int i = 0; i < NDIM; ++i) {
      newActiveRegions[i * newNumRegions + interval_index] =
        dRegions[i * numRegions + tid];
      newActiveRegionsLength[i * newNumRegions + interval_index] =
        dRegionsLength[i * numRegions + tid];
    }

    dRegionsParentIntegral[interval_index] = dRegionsIntegral[tid];
    dRegionsParentError[interval_index] = dRegionsError[tid];

    for (int i = 0; i < numOfDivisionOnDimension; ++i) {
      newActiveRegionsBisectDim[i * newNumRegions + interval_index] =
        subDividingDimension[tid];
    }
  }
}

template <size_t ndim, bool use_custom = false>
class Sub_regions_filter {
public:
  using Regions = Sub_regions<ndim>;
  using Region_char = Region_characteristics<ndim>;
  using Region_ests = Region_estimates<ndim>;

  Sub_regions_filter(const size_t num_regions)
  {
    scanned_array = cuda_malloc<int>(num_regions);
  }

  size_t
  get_num_active_regions(int* active_regions, const size_t num_regions)
  {
    exclusive_scan<int, use_custom>(active_regions, num_regions, scanned_array);
    int last_element;
    int num_active = 0;

    cudaMemcpy(&last_element,
               active_regions + num_regions - 1,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(&num_active,
               scanned_array + num_regions - 1,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    if (last_element == 1)
      num_active++;

    return static_cast<size_t>(num_active);
  }

  // filter out finished regions
  size_t
  filter(Regions& sub_regions,
         Region_char& region_characteristics,
         const Region_ests& region_ests,
         Region_ests& parent_ests)
  {

    const size_t current_num_regions = sub_regions.size;
    const size_t num_active_regions = get_num_active_regions(
      region_characteristics.active_regions, current_num_regions);

    if (num_active_regions == 0) {
      return 0;
    }

    // I dont' create Regions filtered_regions, because upon destruction it
    // would deallocate and for performance reasons, I don't want a deep_copy to
    // occur here
    double* filtered_leftCoord = cuda_malloc<double>(num_active_regions * ndim);
    double* filtered_length = cuda_malloc<double>(num_active_regions * ndim);
    int* filtered_sub_dividing_dim = cuda_malloc<int>(num_active_regions);

    parent_ests.reallocate(num_active_regions);
    const int numOfDivisionOnDimension = 1;
    const size_t num_blocks = compute_num_blocks(current_num_regions);

    alignRegions<double, static_cast<int>(ndim)>
      <<<num_blocks, BLOCK_SIZE>>>(sub_regions.dLeftCoord,
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

    cudaDeviceSynchronize();
    cudaFree(sub_regions.dLeftCoord);
    cudaFree(sub_regions.dLength);
    cudaFree(region_characteristics.sub_dividing_dim);
    sub_regions.dLeftCoord = filtered_leftCoord;
    sub_regions.dLength = filtered_length;
    region_characteristics.sub_dividing_dim = filtered_sub_dividing_dim;
    sub_regions.size = num_active_regions;
    region_characteristics.size = num_active_regions;
    quad::CudaCheckError();
    return num_active_regions;
  }

  size_t
  compute_num_blocks(const size_t num_regions) const
  {
    size_t numThreads = BLOCK_SIZE;
    return num_regions / numThreads + ((num_regions % numThreads) ? 1 : 0);
  }

  ~Sub_regions_filter() { cudaFree(scanned_array); }

  int* scanned_array = nullptr;
};

#endif