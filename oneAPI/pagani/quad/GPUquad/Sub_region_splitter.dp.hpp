#ifndef SUB_REGION_SPLITTER_CUH
#define SUB_REGION_SPLITTER_CUH

#include <CL/sycl.hpp>
#include "oneAPI/pagani/quad/GPUquad/Sub_regions.dp.hpp"
#include "common/oneAPI/cudaMemoryUtil.h"
#include "oneAPI/pagani/quad/GPUquad/heuristic_classifier.dp.hpp"

template <typename T, int NDIM>
void
divideIntervalsGPU(T* genRegions,
                   T* genRegionsLength,
                   T* activeRegions,
                   T* activeRegionsLength,
                   int* activeRegionsBisectDim,
                   size_t numActiveRegions,
                   int numOfDivisionOnDimension,
                   sycl::nd_item<3> item_ct1)
{

  size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
               item_ct1.get_local_id(2);
  if (tid < numActiveRegions) {

    int bisectdim = activeRegionsBisectDim[tid];
    size_t data_size = numActiveRegions * numOfDivisionOnDimension;

    for (int i = 0; i < numOfDivisionOnDimension; ++i) {
      for (int dim = 0; dim < NDIM; ++dim) {
        genRegions[i * numActiveRegions + dim * data_size + tid] =
          activeRegions[dim * numActiveRegions + tid];
        genRegionsLength[i * numActiveRegions + dim * data_size + tid] =
          activeRegionsLength[dim * numActiveRegions + tid];
      }
    }

    for (int i = 0; i < numOfDivisionOnDimension; ++i) {

      T interval_length =
        activeRegionsLength[bisectdim * numActiveRegions + tid] /
        numOfDivisionOnDimension;

      genRegions[bisectdim * data_size + i * numActiveRegions + tid] =
        activeRegions[bisectdim * numActiveRegions + tid] + i * interval_length;
      genRegionsLength[i * numActiveRegions + bisectdim * data_size + tid] =
        interval_length;
    }
  }
}

/*
template<size_t ndim>
void split(Sub_regions<ndim>& sub_regions, const Region_characteristics<ndim>&
classifiers){ size_t num_regions = classifiers.size; size_t num_threads =
BLOCK_SIZE; size_t success = false; size_t num_blocks = num_regions /
num_threads + ((num_regions % num_threads) ? 1 : 0);

    size_t children_per_region = 2;

    double* children_left_coord = quad::cuda_malloc<double>(num_regions * ndim *
children_per_region); double* children_length =
quad::cuda_malloc<double>(num_regions * ndim * children_per_region);

    quad::divideIntervalsGPU<double, ndim>
        <<<num_blocks, num_threads>>>(children_left_coord,
                                      children_length,
                                      sub_regions.dLeftCoord,
                                      sub_regions.dLength,
                                      classifiers.sub_dividing_dim,
                                      num_regions,
                                      children_per_region);
    cudaDeviceSynchronize();
    cudaFree(sub_regions.dLeftCoord);
    cudaFree(sub_regions.dLength);
    sub_regions.size = num_regions * children_per_region;
    sub_regions.dLeftCoord = children_left_coord;
    sub_regions.dLength = children_length;
    quad::CudaCheckError();
}
*/
template <size_t ndim>
class Sub_region_splitter {

public:
  size_t num_regions;
  Sub_region_splitter(size_t size) : num_regions(size) {}

  void
  split(Sub_regions<ndim>* sub_regions,
        const Region_characteristics<ndim>* classifiers)
  {

    auto q_ct1 = sycl::queue(sycl::gpu_selector());
    if (num_regions == 0)
      return;

    size_t num_threads = BLOCK_SIZE;
    size_t num_blocks =
      num_regions / num_threads + ((num_regions % num_threads) ? 1 : 0);
    size_t children_per_region = 2;

    double* children_left_coord =
      quad::cuda_malloc<double>(num_regions * ndim * children_per_region);
    double* children_length =
      quad::cuda_malloc<double>(num_regions * ndim * children_per_region);

    auto dLeftCoord = sub_regions->dLeftCoord;
    auto dLength = sub_regions->dLength;
    auto sub_dividing_dim = classifiers->sub_dividing_dim;

    q_ct1
      .submit([&](sycl::handler& cgh) {
        auto num_regions_ct5 = num_regions;

        cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, num_blocks) *
                                          sycl::range(1, 1, num_threads),
                                        sycl::range(1, 1, num_threads)),
                         [=](sycl::nd_item<3> item_ct1) {
                           divideIntervalsGPU<double, ndim>(children_left_coord,
                                                            children_length,
                                                            dLeftCoord,
                                                            dLength,
                                                            sub_dividing_dim,
                                                            num_regions_ct5,
                                                            children_per_region,
                                                            item_ct1);
                         });
      })
      .wait();

    // dev_ct1.queues_wait_and_throw();
    sycl::free(sub_regions->dLeftCoord, q_ct1);
    sycl::free(sub_regions->dLength, q_ct1);
    sub_regions->size = num_regions * children_per_region;
    sub_regions->dLeftCoord = children_left_coord;
    sub_regions->dLength = children_length;
    quad::CudaCheckError();
  }
};

#endif
