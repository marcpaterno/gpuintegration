#ifndef ONE_API_SUB_REGION_SPLITTER_H
#define ONE_API_SUB_REGION_SPLITTER_H
#include <CL/sycl.hpp>
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/util/MemoryUtil.h"
#include "oneAPI/quad/heuristic_classifier.h"

template <size_t ndim>
class Sub_region_splitter {

public:
  size_t num_regions;
  Sub_region_splitter(sycl::queue& q, size_t size) : num_regions(size) {}

  void split(sycl::queue& q,
             Sub_regions<ndim>& sub_regions,
             const Region_characteristics<ndim>& classifiers);

  template <typename T>
  void divideIntervalsGPU(sycl::queue& q,
                          T* genRegions,
                          T* genRegionsLength,
                          T* activeRegions,
                          T* activeRegionsLength,
                          int* activeRegionsBisectDim,
                          size_t numActiveRegions,
                          int numOfDivisionOnDimension);
};

template <size_t ndim>
void
Sub_region_splitter<ndim>::split(
  sycl::queue& q,
  Sub_regions<ndim>& sub_regions,
  const Region_characteristics<ndim>& classifiers)
{

  size_t num_regions = classifiers.size;

  if (num_regions == 0)
    return;

  size_t num_threads = 64;
  size_t success = false;
  size_t num_blocks =
    num_regions / num_threads + ((num_regions % num_threads) ? 1 : 0);
  // check if zero and return if that's the case
  size_t children_per_region = 2;

  double* children_left_coord =
    sycl::malloc_device<double>(num_regions * ndim * children_per_region, q);
  double* children_length =
    sycl::malloc_device<double>(num_regions * ndim * children_per_region, q);
  // std::cout<<"before divideIntervalsGPU\n";
  divideIntervalsGPU<double>(q,
                             children_left_coord,
                             children_length,
                             sub_regions.dLeftCoord,
                             sub_regions.dLength,
                             classifiers.sub_dividing_dim,
                             num_regions,
                             children_per_region);
  sycl::free(sub_regions.dLeftCoord, q);
  sycl::free(sub_regions.dLength, q);
  sub_regions.size = num_regions * children_per_region;
  sub_regions.dLeftCoord = children_left_coord;
  sub_regions.dLength = children_length;
}

template <size_t ndim>
template <typename T>
void
Sub_region_splitter<ndim>::divideIntervalsGPU(sycl::queue& q,
                                              T* genRegions,
                                              T* genRegionsLength,
                                              T* activeRegions,
                                              T* activeRegionsLength,
                                              int* activeRegionsBisectDim,
                                              size_t numActiveRegions,
                                              int numOfDivisionOnDimension)
{
  q.submit([&](auto& cgh) {
     cgh.parallel_for(sycl::range<1>(numActiveRegions), [=](sycl::id<1> tid) {
       int bisectdim = activeRegionsBisectDim[tid];
       size_t data_size = numActiveRegions * numOfDivisionOnDimension;

       for (int i = 0; i < numOfDivisionOnDimension; ++i) {
         for (size_t dim = 0; dim < ndim; ++dim) {
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
           activeRegions[bisectdim * numActiveRegions + tid] +
           i * interval_length;

         genRegionsLength[i * numActiveRegions + bisectdim * data_size + tid] =
           interval_length;
       }
     });
   })
    .wait();
}

#endif
