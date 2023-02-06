#ifndef KOKKOS_SUB_REGION_FILTER_CUH
#define KOKKOS_SUB_REGION_FILTER_CUH

#include "kokkos/pagani/quad/GPUquad/Sub_regions.cuh"
#include "common/kokkos/cudaMemoryUtil.h"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
#include "common/kokkos/util.cuh"

template <typename T, size_t ndim, bool use_custom = false>
class Sub_regions_filter {
public:
  using Regions = Sub_regions<T, ndim>;
  using Region_char = Region_characteristics<ndim>;
  using Region_ests = Region_estimates<T, ndim>;

  Sub_regions_filter(const size_t num_regions)
  {
    scanned_array = quad::cuda_malloc<int>(num_regions);
  }

  size_t
  get_num_active_regions(ViewVectorInt active_regions, const size_t num_regions)
  {
    exclusive_prefix_scan(active_regions, scanned_array);
    int last_element = -1;
    int num_active = 0;
    int lastScanned = 0.;

    ReturnLastIndexValues(
      active_regions, scanned_array, last_element, lastScanned);
    num_active = static_cast<size_t>(lastScanned);

    if (last_element == 1) {
      num_active++;
    }

    return static_cast<size_t>(num_active);
  }

  void
  ReturnLastIndexValues(ViewVectorInt listA,
                        ViewVectorInt listB,
                        int& lastA,
                        int& lastB)
  {
    int sizeA = listA.extent(0);
    int sizeB = listB.extent(0);

    ViewVectorInt A_sub(listA, std::make_pair(sizeA - 1, sizeA));
    ViewVectorInt B_sub(listB, std::make_pair(sizeB - 1, sizeB));

    ViewVectorInt::HostMirror hostA_sub = Kokkos::create_mirror_view(A_sub);
    ViewVectorInt::HostMirror hostB_sub = Kokkos::create_mirror_view(B_sub);

    deep_copy(hostA_sub, A_sub);
    deep_copy(hostB_sub, B_sub);

    lastA = hostA_sub(0);
    lastB = hostB_sub(0);
  }

  void
  alignRegions(ViewVectorDouble dRegions,
               ViewVectorDouble dRegionsLength,
               ViewVectorInt activeRegions,
               ViewVectorDouble dRegionsIntegral,
               ViewVectorDouble dRegionsError,
               ViewVectorDouble dRegionsParentIntegral,
               ViewVectorDouble dRegionsParentError,
               ViewVectorInt subDividingDimension,
               ViewVectorInt scannedArray,
               ViewVectorDouble newActiveRegions,
               ViewVectorDouble newActiveRegionsLength,
               ViewVectorInt newActiveRegionsBisectDim,
               size_t numRegions,
               size_t newNumRegions,
               size_t numOfDivisionOnDimension)
  {
    size_t numThreads = 64;
    size_t numBlocks =
      numRegions / numThreads + ((numRegions % numThreads) ? 1 : 0);

    Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> team_policy1(numBlocks,
                                                                  numThreads);
    auto team_policy = Kokkos::Experimental::require(
      team_policy1, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for(
      "AlignRegions",
      team_policy,
      KOKKOS_LAMBDA(const member_type team_member) {
        int threadIdx = team_member.team_rank();
        int blockIdx = team_member.league_rank();

        size_t tid = blockIdx * numThreads + threadIdx;

        if (tid < numRegions && activeRegions(tid) == 1.) {
          size_t interval_index = (size_t)scannedArray(tid);

          for (size_t i = 0; i < ndim; ++i) {
            newActiveRegions(i * newNumRegions + interval_index) =
              dRegions(i * numRegions + tid);
            newActiveRegionsLength(i * newNumRegions + interval_index) =
              dRegionsLength(i * numRegions + tid);
          }

          dRegionsParentIntegral(interval_index) = dRegionsIntegral(tid);
          dRegionsParentError(interval_index) = dRegionsError(tid);

          for (size_t i = 0; i < numOfDivisionOnDimension; ++i) {
            newActiveRegionsBisectDim(i * newNumRegions + interval_index) =
              subDividingDimension(tid);
          }
        }
      });
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
    ViewVectorDouble filtered_leftCoord =
      quad::cuda_malloc<T>(num_active_regions * ndim);
    ViewVectorDouble filtered_length =
      quad::cuda_malloc<T>(num_active_regions * ndim);
    ViewVectorInt filtered_sub_dividing_dim =
      quad::cuda_malloc<int>(num_active_regions);

    parent_ests.reallocate(num_active_regions);
    const int numOfDivisionOnDimension = 1;

    alignRegions(sub_regions.dLeftCoord,
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

    sub_regions.dLeftCoord = filtered_leftCoord;
    sub_regions.dLength = filtered_length;
    region_characteristics.sub_dividing_dim = filtered_sub_dividing_dim;
    sub_regions.size = num_active_regions;
    region_characteristics.size = num_active_regions;
    return num_active_regions;
  }

  ~Sub_regions_filter() {}

  ViewVectorInt scanned_array;
};

#endif