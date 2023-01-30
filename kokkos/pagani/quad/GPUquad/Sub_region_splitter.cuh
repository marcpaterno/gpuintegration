#ifndef SUB_REGION_SPLITTER_CUH
#define SUB_REGION_SPLITTER_CUH

#include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "common/cuda/cudaMemoryUtil.h"
#include "cuda/pagani/quad/GPUquad/heuristic_classifier.cuh"

template <typename T, size_t ndim>
class Sub_region_splitter {

public:
  size_t num_regions;
  Sub_region_splitter(size_t size) : num_regions(size) {}

  void
  split(Sub_regions<T, ndim>& sub_regions,
        const Region_characteristics<ndim>& classifiers)
  {
    if (num_regions == 0)
      return;

    size_t num_threads = BLOCK_SIZE;
    size_t num_blocks =
      num_regions / num_threads + ((num_regions % num_threads) ? 1 : 0);
    size_t children_per_region = 2;

    ViewVectorDouble children_left_coord(
      "children_left", num_regions * ndim * children_per_region);
    ViewVectorDouble children_length("children_length",
                                     num_regions * ndim * children_per_region);

    divideIntervalsGPU(children_left_coord,
                       children_length,
                       sub_regions.dLeftCoord,
                       sub_regions.dLength,
                       classifiers.sub_dividing_dim,
                       num_regions,
                       children_per_region);

    // is the old sub_regions.dLeftCoord getting free?
    // is the old sub_regions.dLength getting free?

    sub_regions.size = num_regions * children_per_region;
    sub_regions.dLeftCoord = children_left_coord;
    sub_regions.dLength = children_length;
  }

  void
  divideIntervalsGPU(T* genRegions,
                     T* genRegionsLength,
                     T* activeRegions,
                     T* activeRegionsLength,
                     int* activeRegionsBisectDim,
                     size_t numActiveRegions,
                     int numOfDivisionOnDimension)
  {
    size_t numThreads = 64;
    size_t numBlocks =
      numActiveRegions / numThreads + ((numActiveRegions % numThreads) ? 1 : 0);
    Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> team_policy1(numBlocks,
                                                                  numThreads);
    auto team_policy = Kokkos::Experimental::require(
      team_policy1, Kokkos::Experimental::WorkItemProperty::HintLightWeight);

    Kokkos::parallel_for(
      "DivideIntervalsGPU",
      team_policy,
      KOKKOS_LAMBDA(const member_type team_member) {
        int threadIdx = team_member.team_rank();
        int blockIdx = team_member.league_rank();
        size_t tid = blockIdx * numThreads + threadIdx;

        if (tid < numActiveRegions) {

          int bisectdim = activeRegionsBisectDim(tid);
          size_t data_size = numActiveRegions * numOfDivisionOnDimension;

          for (int i = 0; i < numOfDivisionOnDimension; ++i) {
            for (int dim = 0; dim < ndim; ++dim) {
              genRegions(i * numActiveRegions + dim * data_size + tid) =
                activeRegions(dim * numActiveRegions + tid);
              genRegionsLength(i * numActiveRegions + dim * data_size + tid) =
                activeRegionsLength(dim * numActiveRegions + tid);
            }
          }

          for (int i = 0; i < numOfDivisionOnDimension; ++i) {

            double interval_length =
              activeRegionsLength(bisectdim * numActiveRegions + tid) /
              numOfDivisionOnDimension;

            genRegions(bisectdim * data_size + i * numActiveRegions + tid) =
              activeRegions(bisectdim * numActiveRegions + tid) +
              i * interval_length;
            genRegionsLength(i * numActiveRegions + bisectdim * data_size +
                             tid) = interval_length;
          }
        }
      });
  }
};
#endif