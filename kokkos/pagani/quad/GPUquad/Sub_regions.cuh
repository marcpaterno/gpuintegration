#ifndef SUB_REGIONS_CUH
#define SUB_REGIONS_CUH

#include <iostream>
#include "common/kokkos/cudaMemoryUtil.h"
#include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "common/kokkos/Volume.cuh"

template <typename T, size_t ndim>
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

  Sub_regions(const Sub_regions<T,ndim>& other)
  {
    device_init(other.size);
	dLeftCoord = other.dLeftCoord;
	dLength = other.dLength;
	//Kokkos::deep_copy(dLeftCoord, other.dLeftCoord);
	//Kokkos::deep_copy(dLength, other.dLength);
  }


  ~Sub_regions()
  {}
	
	void
	create_uniform_split(size_t numOfDivisionPerRegionPerDimension)
	{
		size_t num_starting_regions = pow((double)numOfDivisionPerRegionPerDimension, (double)ndim);
		double starting_axis_length = 1./(double)numOfDivisionPerRegionPerDimension;
		device_init(num_starting_regions);
		Kokkos::parallel_for(
		  "GenerateInitialRegions",
		  Kokkos::RangePolicy<>(0, num_starting_regions),
			[=,*this] __host__ __device__(const int reg) {
				
				for(int dim = 0; dim < (int)ndim; ++dim){				
					size_t _id = (int)(reg / pow((double)numOfDivisionPerRegionPerDimension, dim)) % numOfDivisionPerRegionPerDimension;
					dLeftCoord[num_starting_regions * dim + reg] = static_cast<double>(_id) * static_cast<double>(starting_axis_length);  
					dLength[num_starting_regions * dim + reg] = starting_axis_length;
				}
		  });
		size = num_starting_regions;
	}	
	
  void
  device_init(size_t const numRegions)
  {
    size = numRegions;
    dLeftCoord = quad::cuda_malloc<T>(numRegions * ndim);
    dLength = quad::cuda_malloc<T>(numRegions * ndim);
  }

  void
  print_bounds()
  {
    auto LeftCoord = Kokkos::create_mirror_view(dLeftCoord);
    auto Length = Kokkos::create_mirror_view(dLength);
	
	Kokkos::deep_copy(dLeftCoord, LeftCoord);
	Kokkos::deep_copy(dLength, Length);

	for (size_t i = 0; i < size; i++) {
      for (size_t dim = 0; dim < ndim; dim++) {
        printf("region %lu, %lu, %f, %f, %f",
               i,
               dim,
               LeftCoord[size * dim + i],
               LeftCoord[size * dim + i] + Length[size * dim + i],
               Length[size * dim + i]);
      }
      printf("\n");
    }
  }

  T
  compute_region_volume(size_t region_id, double* region_lengths, size_t nregions)
  {
    T reg_vol = 1.;
    for (size_t dim = 0; dim < ndim; dim++) {
      size_t region_index = size * dim + region_id;
		
      reg_vol *= region_lengths[region_index];
    }
    return reg_vol;
  }

  T
  compute_total_volume()
  {
	  
	printf("about to compute_total_volume\n");
	auto LeftCoord = Kokkos::create_mirror_view(dLeftCoord);
    auto Length = Kokkos::create_mirror_view(dLength);
	
	Kokkos::deep_copy(LeftCoord, dLeftCoord);
	Kokkos::deep_copy(Length, dLength);    

    T total_vol = 0.;
    for (size_t regID = 0; regID < size; regID++) {
      total_vol += compute_region_volume(regID, Length.data(), Length.extent(0));
    }

    return total_vol;
  }

  void
  uniform_split(size_t numOfDivisionPerRegionPerDimension)
  {
    size_t num_starting_regions =
      pow((T)numOfDivisionPerRegionPerDimension, (T)ndim);
    T starting_axis_length = 1. / (T)numOfDivisionPerRegionPerDimension;

    device_init(num_starting_regions);

    create_uniform_split(numOfDivisionPerRegionPerDimension);
    size = num_starting_regions;
  }

  quad::Volume<T, ndim>
  extract_region(size_t const regionID)
  {
	auto LeftCoord = Kokkos::create_mirror_view(dLeftCoord);
    auto Length = Kokkos::create_mirror_view(dLength);
	
    quad::Volume<T, ndim> regionID_bounds;
    for (size_t dim = 0; dim < ndim; dim++) {
      size_t region_index = size * dim + regionID;
      regionID_bounds.lows[dim] = LeftCoord[region_index];
      regionID_bounds.highs[dim] =
        LeftCoord[region_index] + Length[region_index];
    }
    return regionID_bounds;
  }

  /*void
  set_ptr_to_estimates(Region_estimates<T, ndim>* estimates)
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
*/
  void
  take_snapshot()
  {
    snapshot_size = size;
    snapshot_dLeftCoord = quad::cuda_malloc<T>(size * ndim);
    snapshot_dLength = quad::cuda_malloc<T>(size * ndim);
    quad::cuda_memcpy_device_to_device<T>(
      snapshot_dLeftCoord, dLeftCoord, size * ndim);
    quad::cuda_memcpy_device_to_device<T>(snapshot_dLength, dLength, size * ndim);
  }

  void
  load_snapshot()
  {
    dLeftCoord = snapshot_dLeftCoord;
    dLength = snapshot_dLength;
    size = snapshot_size;
  }

  // device side variables
  ViewVectorDouble dLeftCoord;
  ViewVectorDouble dLength;

  ViewVectorDouble snapshot_dLeftCoord;
  ViewVectorDouble snapshot_dLength;
  Region_characteristics<ndim>* characteristics;
  Region_estimates<T, ndim>* region_estimates;

  size_t size = 0;
  size_t host_data_size = 0;
  size_t snapshot_size = 0;
};

#endif