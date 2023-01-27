#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "common/cuda/cudaMemoryUtil.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <array>

double 
compute_jacobian(double* left_coord, double* length, size_t size){
    double jacobian = 1.;
    for(size_t i=0; i < size; ++i)
        jacobian *= length[i];
    return jacobian;
}

TEST_CASE("Split 3D space in unit-hypercube"){
    constexpr size_t ndim = 3;
    size_t partitions_per_axis = 2;   
    Sub_regions<double, ndim> sub_regions(partitions_per_axis);    
    
    sub_regions.host_init();
    sub_regions.refresh_host_device();
    
    for(int reg = 0; reg < sub_regions.size; ++reg){
        double reg_vol = sub_regions.compute_region_volume(reg);
        CHECK(reg_vol < 1.);
        CHECK(reg_vol > 0.);
        CHECK(reg_vol == Approx(1./static_cast<double>(sub_regions.size)));
    }
    
    CHECK(sub_regions.compute_total_volume() == 1.);
    
    for(int i=0; i < ndim * sub_regions.size; ++i){
        CHECK(sub_regions.LeftCoord[i] >= 0.);
        CHECK(sub_regions.Length[i] < 1.);
        CHECK(sub_regions.Length[i] > 0.);
    }
}

TEST_CASE("Split 5D space in unit-hypercube"){
    constexpr size_t ndim = 5;
    size_t partitions_per_axis = 3;   
    Sub_regions<double, ndim> sub_regions(partitions_per_axis);    
    
    sub_regions.host_init();
    sub_regions.refresh_host_device();
    
    for(int reg = 0; reg < sub_regions.size; ++reg){
        double reg_vol = sub_regions.compute_region_volume(reg);
        CHECK(reg_vol < 1.);
        CHECK(reg_vol > 0.);
        CHECK(reg_vol == Approx(1./static_cast<double>(sub_regions.size)));
    }
    
    CHECK(sub_regions.compute_total_volume() == 1.);
    
    for(int i=0; i < ndim * sub_regions.size; ++i){
        CHECK(sub_regions.LeftCoord[i] >= 0.);
        CHECK(sub_regions.Length[i] < 1.);
        CHECK(sub_regions.Length[i] > 0.);
    }
}

TEST_CASE("Split 10D space in unit-hypercube"){
    constexpr size_t ndim = 8;
	size_t partitions_per_axis = 4;   
    Sub_regions<double, ndim> sub_regions(partitions_per_axis);    
    
    sub_regions.host_init();
    sub_regions.refresh_host_device();
    
    for(int reg = 0; reg < sub_regions.size; ++reg){
        double reg_vol = sub_regions.compute_region_volume(reg);
        CHECK(reg_vol < 1.);
        CHECK(reg_vol > 0.);
        CHECK(reg_vol == Approx(1./static_cast<double>(sub_regions.size)));
    }
    
    CHECK(sub_regions.compute_total_volume() == 1.);
    
    for(int i=0; i < ndim * sub_regions.size; ++i){
        CHECK(sub_regions.LeftCoord[i] >= 0.);
        CHECK(sub_regions.Length[i] < 1.);
        CHECK(sub_regions.Length[i] > 0.);
    }
}