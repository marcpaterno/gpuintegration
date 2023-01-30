#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/demos/function.cuh"
#include "cuda/pagani/quad/GPUquad/Sample.cuh"
#include "cuda/pagani/quad/quad.h"
#include "common/cuda/cudaMemoryUtil.h"
#include "common/cuda/Volume.cuh"
#include "common/cuda/cudaUtil.h"
#include "common/cuda/custom_functions.cuh"
#include "common/cuda/thrust_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "cuda/pagani/quad/GPUquad/Sub_region_splitter.cuh"
#include "cuda/pagani/quad/GPUquad/Region_characteristics.cuh"

#include "common/cuda/integrands.cuh"
#include "common/integration_result.hh"

template<size_t ndim>
bool is_free_of_duplicates(Sub_regions<double, ndim>& regions){
    for(size_t regionID = 0; regionID < regions.size; regionID++){
        quad::Volume<double, ndim> region = regions.extract_region(regionID);
        for(size_t reg = 0; reg < regions.size; reg++){
            quad::Volume<double, ndim> region_i = regions.extract_region(reg);
            if(reg != regionID && region == region_i){
                return false;
            }
        }
    }
    return true;
}

TEST_CASE("Split all regions at dim 1")
{
	constexpr int ndim = 2;
	Sub_regions<double, ndim> regions(5);
	const size_t n = regions.size;
	
	Sub_region_splitter<double, ndim> splitter(n);
	Region_characteristics<ndim> classifications(n);
	
	int* sub_div_dim = quad::host_alloc<int>(n);
	double* orig_leftcoord = quad::host_alloc<double>(n*ndim);
	double* orig_length = quad::host_alloc<double>(n*ndim);
	
	quad::cuda_memcpy_to_host<double>(orig_leftcoord, regions.dLeftCoord, n*ndim);
	quad::cuda_memcpy_to_host<double>(orig_length, regions.dLength, n*ndim);
	
	for(int i = 0; i < n; ++i){
		sub_div_dim[i] = 1;
	}
	
	quad::cuda_memcpy_to_device<int>(classifications.sub_dividing_dim, sub_div_dim, n);
	splitter.split(regions, classifications);
	
	double* new_leftcoord = quad::host_alloc<double>(2*n*ndim);
	double* new_length = quad::host_alloc<double>(2*n*ndim);
	
	quad::cuda_memcpy_to_host<double>(new_leftcoord, regions.dLeftCoord, n*2*ndim);
	quad::cuda_memcpy_to_host<double>(new_length, regions.dLength, n*2*ndim);
	SECTION("Dimension zero is intact")
	{
		for(int i=0; i < n; ++i){
			const size_t dim = 0;
			const size_t par_index = i + dim*n;
			const size_t left =  i + dim*n*2;
			const size_t right = n + i + dim*n*2;
			
			CHECK(new_leftcoord[left] == Approx(orig_leftcoord[par_index]));
			CHECK(new_leftcoord[right] == Approx(orig_leftcoord[par_index]));

			CHECK(new_length[left] == Approx(orig_length[par_index]));
			CHECK(new_length[right] == Approx(orig_length[par_index]));
		}
	}
		
	SECTION("Dimension one is changed")
	{
		for(int i=0; i < n; ++i){
			const size_t dim = 1;
			const size_t par_index = i + dim*n;
			const size_t left =  i + dim*n*2;
			const size_t right = n + i + dim*n*2;
			
			CHECK(new_leftcoord[left] == Approx(orig_leftcoord[par_index]));
			CHECK(new_leftcoord[right] == Approx(orig_leftcoord[par_index] + orig_length[par_index]/2));

			CHECK(new_length[left] == Approx(orig_length[par_index]/2));
			CHECK(new_length[right] == Approx(orig_length[par_index]/2));
		}
	}
	
	delete[] new_length;
	delete[] new_leftcoord;
	delete[] sub_div_dim;
}