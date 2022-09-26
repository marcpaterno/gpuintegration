#ifndef ONE_API_SUB_REGIONS_H
#define ONE_API_SUB_REGIONS_H

#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/Region_characteristics.h"
#include "oneAPI/quad/util/Volume.h"
#include "oneAPI/quad/util/MemoryUtil.h"

template <size_t ndim>
struct Sub_regions {
  Sub_regions() {}
  Sub_regions(sycl::queue& q, const size_t partitions_per_axis);
  ~Sub_regions();

  void host_device_init(sycl::queue& q, const size_t numRegions);
  void print_bounds();
  double compute_region_volume(size_t const regionID);
  double compute_total_volume();
  void uniform_split(sycl::queue& q, size_t numOfDivisionPerRegionPerDimension);
  // quad::Volume<double, ndim> extract_region(size_t const regionID);

  double* dLeftCoord = nullptr;
  double* dLength = nullptr;
  size_t size = 0;
  sycl::queue* _q;
};

template <size_t ndim>
Sub_regions<ndim>::Sub_regions(sycl::queue& q, const size_t partitions_per_axis)
{
  uniform_split(q, partitions_per_axis);
  _q = &q;
}

template <size_t ndim>
Sub_regions<ndim>::~Sub_regions()
{
  sycl::free(dLeftCoord, *_q);
  sycl::free(dLength, *_q);
}

template <size_t ndim>
void
Sub_regions<ndim>::host_device_init(sycl::queue& q, const size_t numRegions)
{
  dLeftCoord = sycl::malloc_device<double>(numRegions * ndim, q);
  dLength = sycl::malloc_device<double>(numRegions * ndim, q);

  size = numRegions;
}

template <size_t ndim>
void
Sub_regions<ndim>::print_bounds()
{

  double* LeftCoord = quad::allocate_and_copy_to_host<double>(dLeftCoord, size);
  double* Length = quad::allocate_and_copy_to_host<double>(dLength, size);

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

template <size_t ndim>
double
Sub_regions<ndim>::compute_region_volume(size_t const regionID)
{
  double reg_vol = 1.;
  for (size_t dim = 0; dim < ndim; dim++) {
    size_t region_index = size * dim + regionID;
    reg_vol *= dLength[region_index];
  }

  return reg_vol;
}

template <size_t ndim>
double
Sub_regions<ndim>::compute_total_volume()
{
  double total_vol = 0.;
  for (size_t regID = 0; regID < size; regID++) {
    total_vol += compute_region_volume(regID);
  }

  return total_vol;
}

/*template<size_t ndim>
quad::Volume<double, ndim>
 Sub_regions<ndim>::extract_region(size_t const regionID){

  quad::Volume<double, ndim> regionID_bounds;
  for(size_t dim = 0; dim < ndim; dim++){
    size_t region_index = size * dim + regionID;
    regionID_bounds.lows[dim] = dLeftCoord[region_index];
    regionID_bounds.highs[dim] = dLeftCoord[region_index] +
dLength[region_index];
  }
  return regionID_bounds;
}*/

template <size_t ndim>
void
Sub_regions<ndim>::uniform_split(sycl::queue& q, size_t _divs_per_d)
{
  // int divs_per_dim = divs_per_d;
  const int nregions = pow(_divs_per_d, ndim);
  const double length = 1. / (double)_divs_per_d;
  dLeftCoord = sycl::malloc_device<double>(nregions * ndim, q);
  dLength = sycl::malloc_device<double>(nregions * ndim, q);

  // compiler does not allow me to capture *this in lambda below, error:
  // static_assert type not device copyable
  double* LeftCoord = dLeftCoord;
  double* Length = dLength;

  size_t default_num_threads = 512;
  size_t numBlocks =
    (size_t)ceil((double)nregions / (double)default_num_threads);

  q.submit([&](auto& cgh) {
     sycl::stream str(262144, 4096, cgh);
     cgh.parallel_for(sycl::range<1>(nregions), [=](sycl::id<1> _reg) {
       size_t reg = _reg[0];

       int divs_per_dim = 2;
       if (ndim < 5)
         divs_per_dim = 4;
       else if (ndim <= 11)
         divs_per_dim = 2;
       else
         divs_per_dim = 1;

       for (size_t dim = 0; dim < ndim; ++dim) {
         int _id = (int)(reg / (int)pow(divs_per_dim, dim)) % divs_per_dim;
         LeftCoord[nregions * dim + (reg)] = (double)(_id)*length;
         Length[nregions * dim + (reg)] = length;
       }
     });
   })
    .wait();

  size = nregions;
}
#endif
