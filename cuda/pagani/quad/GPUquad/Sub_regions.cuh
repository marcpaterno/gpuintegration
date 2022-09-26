#ifndef SUB_REGIONS_CUH
#define SUB_REGIONS_CUH

#include <iostream>
#include "cuda/pagani/quad/util/mem_util.cuh"
#include "cuda/pagani/quad/GPUquad/Region_estimates.cuh"
#include "cuda/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "cuda/pagani/quad/util/Volume.cuh"
#include <cuda.h>

__global__ void
create_uniform_split(const double length,
                     double* newRegions,
                     double* newRegionsLength,
                     const size_t newNumOfRegions,
                     const size_t numOfDivisionsPerRegionPerDimension,
                     size_t ndim)
{
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId < newNumOfRegions) {
    size_t interval_index =
      threadId / pow((double)numOfDivisionsPerRegionPerDimension, (double)ndim);
    size_t local_id =
      threadId %
      (size_t)pow((double)numOfDivisionsPerRegionPerDimension, (double)ndim);
    for (int dim = 0; dim < ndim; ++dim) {
      size_t id =
        (size_t)(local_id / pow((double)numOfDivisionsPerRegionPerDimension,
                                (double)dim)) %
        numOfDivisionsPerRegionPerDimension;
      newRegions[newNumOfRegions * dim + threadId] = id * length;
      newRegionsLength[newNumOfRegions * dim + threadId] = length;
    }
  }
}

template <size_t ndim>
struct Sub_regions {

  // constructor should probably just allocate
  // not partition the axis, that should be turned to a specific method instead
  // for clarity, current way is counter-intuitive since the other sub-region
  // related structs allocate with their constructors
  Sub_regions() {}

  Sub_regions(const size_t partitions_per_axis)
  {
    uniform_split(partitions_per_axis);
  }

  ~Sub_regions()
  {
    delete[] LeftCoord;
    delete[] Length;
    cudaFree(dLeftCoord);
    cudaFree(dLength);
  }

  void
  host_device_init(const size_t numRegions)
  {
    LeftCoord = host_alloc<double>(numRegions * ndim);
    Length = host_alloc<double>(numRegions * ndim);

    dLeftCoord = cuda_malloc<double>(numRegions * ndim);
    dLength = cuda_malloc<double>(numRegions * ndim);

    size = numRegions;
    host_data_size = numRegions;
  }

  void
  refresh_host_device()
  {
    cuda_memcpy_to_host<double>(LeftCoord, dLeftCoord, size * ndim);
    cuda_memcpy_to_host<double>(Length, dLength, size * ndim);
  }

  void
  refresh_device_from_host()
  {
    cuda_memcpy_to_device<double>(dLeftCoord, LeftCoord, size * ndim);
    cuda_memcpy_to_device<double>(dLength, Length, size * ndim);
  }

  void
  host_init()
  {
    /*if(LeftCoord != nullptr || Length != nullptr){
        //std::cout<<"host arrays are not set to null, not allocating\n";
        return;
    }*/
    free(LeftCoord);
    free(Length);
    host_data_size = size;
    LeftCoord = host_alloc<double>(size * ndim);
    Length = host_alloc<double>(size * ndim);
  }

  void
  device_init(size_t const numRegions)
  {
    cudaFree(dLeftCoord);
    cudaFree(dLength);
    size = numRegions;
    dLeftCoord = cuda_malloc<double>(numRegions * ndim);
    dLength = cuda_malloc<double>(numRegions * ndim);
  }

  void
  print_bounds()
  {
    refresh_host_device();
    for (size_t i = 0; i < size; i++) {
      for (size_t dim = 0; dim < ndim; dim++) {
        printf("region %lu, dim %lu, bounds: %f, %f (length:%f)\n",
               i,
               dim,
               LeftCoord[size * dim + i],
               LeftCoord[size * dim + i] + Length[size * dim + i],
               Length[size * dim + i]);
      }
      printf("\n");
    }
  }

  double
  compute_region_volume(size_t const regionID)
  {
    double reg_vol = 1.;
    for (size_t dim = 0; dim < ndim; dim++) {
      size_t region_index = size * dim + regionID;

      reg_vol *= Length[region_index];
    }
    return reg_vol;
  }

  double
  compute_total_volume()
  {
    host_init();
    refresh_host_device();

    double total_vol = 0.;
    for (size_t regID = 0; regID < size; regID++) {
      total_vol += compute_region_volume(regID);
    }

    return total_vol;
  }

  void
  uniform_split(size_t numOfDivisionPerRegionPerDimension)
  {
    size_t num_starting_regions =
      pow((double)numOfDivisionPerRegionPerDimension, (double)ndim);
    double starting_axis_length =
      1. / (double)numOfDivisionPerRegionPerDimension;

    device_init(num_starting_regions);

    size_t numThreads = 512;
    size_t numBlocks =
      (size_t)ceil((double)num_starting_regions / (double)numThreads);

    create_uniform_split<<<numBlocks, numThreads>>>(
      starting_axis_length,
      dLeftCoord,
      dLength,
      num_starting_regions,
      numOfDivisionPerRegionPerDimension,
      ndim);
    cudaDeviceSynchronize();
    size = num_starting_regions;
  }

  quad::Volume<double, ndim>
  extract_region(size_t const regionID)
  {

    if (LeftCoord == nullptr || Length == nullptr) {
      // printf("host_init to be invoked within Sub_regions::extra_region\n");
      host_init();
    }
    refresh_host_device();
    quad::Volume<double, ndim> regionID_bounds;
    for (size_t dim = 0; dim < ndim; dim++) {
      size_t region_index = size * dim + regionID;
      regionID_bounds.lows[dim] = LeftCoord[region_index];
      regionID_bounds.highs[dim] =
        LeftCoord[region_index] + Length[region_index];
    }
    return regionID_bounds;
  }

  void
  set_ptr_to_estimates(Region_estimates<ndim>* estimates)
  {
    assert(estimates != nullptr && estimates->size == this->size);
    region_estimates = estimates;
  }

  void
  set_ptr_to_characteristics(Region_characteristics<ndim>* charactrs)
  {
    assert(charactrs != nullptr && charactrs->size == this->size);
    characteristics = charactrs;
  }

  void
  take_snapshot()
  {
    snapshot_size = size;
    snapshot_dLeftCoord = cuda_malloc<double>(size * ndim);
    snapshot_dLength = cuda_malloc<double>(size * ndim);
    cuda_memcpy_device_to_device<double>(
      snapshot_dLeftCoord, dLeftCoord, size * ndim);
    cuda_memcpy_device_to_device<double>(
      snapshot_dLength, dLength, size * ndim);
  }

  void
  load_snapshot()
  {
    cudaFree(dLeftCoord);
    cudaFree(dLength);
    dLeftCoord = snapshot_dLeftCoord;
    dLength = snapshot_dLength;
    size = snapshot_size;
  }

  // for accessing on the host side, may need to invoke refresh_host_device() to
  // do copy
  double* LeftCoord = nullptr;
  double* Length = nullptr;

  // device side variables
  double* dLeftCoord = nullptr;
  double* dLength = nullptr;

  double* snapshot_dLeftCoord = nullptr;
  double* snapshot_dLength;
  Region_characteristics<ndim>* characteristics;
  Region_estimates<ndim>* region_estimates;

  size_t size = 0;
  size_t host_data_size = 0;
  size_t snapshot_size = 0;
};

#endif