#include "kokkos/pagani/quad/GPUquad/Sub_region_filter.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
#include "catch2/catch.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <array>


TEST_CASE("Detect Number of active regions for list-size of 100")
{
	constexpr int ndim = 2;
	constexpr size_t num_regions = 100;
	Region_characteristics<ndim> regions(num_regions);
	Sub_regions_filter<double, ndim> filter(num_regions);
	
	auto host_active_regions = Kokkos::create_mirror_view(regions.active_regions);
	
	SECTION("All active regions")
	{
		for(size_t  i=0; i < num_regions; ++i)
			host_active_regions[i] = 1;
		
		Kokkos::deep_copy(regions.active_regions, host_active_regions);
		size_t num_active = filter.get_num_active_regions(regions.active_regions, regions.size);
		CHECK(num_active == num_regions);
	}

	SECTION("All but last region is active")
	{
		host_active_regions[regions.size - 1] = 1;
		
		Kokkos::deep_copy(regions.active_regions, host_active_regions);
		size_t num_active = filter.get_num_active_regions(regions.active_regions, regions.size);
		CHECK(num_active == 1);
	}
	
	SECTION("All but the first region is active")
	{
		host_active_regions[0] = 1;
		
		Kokkos::deep_copy(regions.active_regions, host_active_regions);
		size_t num_active = filter.get_num_active_regions(regions.active_regions, regions.size);
		CHECK(num_active == 1);
	}
	
	SECTION("Zero active regions")
	{		
		Kokkos::deep_copy(regions.active_regions, host_active_regions);
		size_t num_active = filter.get_num_active_regions(regions.active_regions, regions.size);
		CHECK(num_active == 0);
	}
	
	SECTION("Only first region is inactive")
	{		
		for(size_t  i=0; i < num_regions; ++i)
			host_active_regions[i] = 1;
		host_active_regions[0] = 0;
		
		Kokkos::deep_copy(regions.active_regions, host_active_regions);
		size_t num_active = filter.get_num_active_regions(regions.active_regions, regions.size);
		CHECK(num_active == 99);
	}
	
		SECTION("Only last region is inactive")
	{		
		for(size_t  i=0; i < num_regions; ++i)
			host_active_regions[i] = 1;
		host_active_regions[regions.size - 1] = 0;
		
		Kokkos::deep_copy(regions.active_regions, host_active_regions);
		size_t num_active = filter.get_num_active_regions(regions.active_regions, regions.size);
		CHECK(num_active == 99);
	}
};

TEST_CASE("Only 1 region to classify as active/inactive")
{
	constexpr int ndim = 2;
	constexpr size_t num_regions = 1;
	Region_characteristics<ndim> regions(num_regions);
	Sub_regions_filter<double, ndim> filter(num_regions);
	
	auto host_active_regions = Kokkos::create_mirror_view(regions.active_regions);
	
	SECTION("one active region")
	{
		host_active_regions[0] = 1;
		
		Kokkos::deep_copy(regions.active_regions, host_active_regions);
		size_t num_active = filter.get_num_active_regions(regions.active_regions, regions.size);
		CHECK(num_active == 1);
	}

	SECTION("zero active region")
	{
		host_active_regions[0] = 0;
		
		Kokkos::deep_copy(regions.active_regions, host_active_regions);
		size_t num_active = filter.get_num_active_regions(regions.active_regions, regions.size);
		CHECK(num_active == 0);
	}
};

TEST_CASE("Filtering with all active regions")
{
	constexpr int ndim = 2;
	Sub_regions<double, ndim> coordinates(10);
	size_t n = coordinates.size;
	Region_characteristics<ndim> classifications(n);
	Sub_regions_filter<double, ndim> filter(n);
	Region_estimates<double, ndim> estimates(n);
	Region_estimates<double, ndim> parents(n/2);

	auto integrals_mirror = Kokkos::create_mirror_view(estimates.integral_estimates);
	auto error_mirror = Kokkos::create_mirror_view(estimates.error_estimates);
	auto parent_integrals_mirror = Kokkos::create_mirror_view(parents.integral_estimates);
	auto parent_error_mirror = Kokkos::create_mirror_view(parents.error_estimates);	
	auto sub_dividing_dim = Kokkos::create_mirror_view(classifications.sub_dividing_dim);	
	auto host_active_regions = Kokkos::create_mirror_view(classifications.active_regions);	
	auto original_LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);	
	auto original_Length = Kokkos::create_mirror_view(coordinates.dLength);	
	
	//set region variabels to fit test scenario
	for(size_t i=0; i < n; ++i){
		integrals_mirror[i] = 1000.;
		error_mirror[i] = 10.;
		host_active_regions[i] = 1;
		sub_dividing_dim[i] = 0;
			
		if(static_cast<size_t>(i) < n/2){
			parent_integrals_mirror[i] = 1500.;
			parent_error_mirror[i] = 20.;
		}
	}
		
	Kokkos::deep_copy(classifications.active_regions, host_active_regions);
	Kokkos::deep_copy(estimates.integral_estimates, integrals_mirror);
	Kokkos::deep_copy(estimates.error_estimates, error_mirror);
	Kokkos::deep_copy(parents.integral_estimates, parent_integrals_mirror);
	Kokkos::deep_copy(parents.error_estimates, parent_error_mirror);
	Kokkos::deep_copy(sub_dividing_dim, classifications.sub_dividing_dim);
	
	size_t num_active = filter.filter(coordinates, classifications, estimates, parents);
	CHECK(num_active == coordinates.size);
	
	//after filter, the pre-filtering integral and error-estimates become the parents
	Kokkos::deep_copy(integrals_mirror, parents.integral_estimates);
	Kokkos::deep_copy(error_mirror, parents.error_estimates);
	
	auto new_subdiv_dim = Kokkos::create_mirror_view(classifications.sub_dividing_dim);	
	auto LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);	
	auto Length = Kokkos::create_mirror_view(coordinates.dLength);	
	
	CHECK(integrals_mirror[16] == 1000.);
	CHECK(integrals_mirror[15] == 1000.);
	CHECK(integrals_mirror[14] == 1000.);
		
	CHECK(error_mirror[16] == 10.);
	CHECK(error_mirror[15] == 10.);
	CHECK(error_mirror[14] == 10.);
		
	CHECK(sub_dividing_dim[16] == 0);
	CHECK(sub_dividing_dim[15] == 0);
	CHECK(sub_dividing_dim[14] == 0);
		
	//check if
	CHECK(LeftCoord[15] == Approx(original_LeftCoord[15]));
	CHECK(Length[15] == Approx(original_Length[15]));
		
	CHECK(LeftCoord[14] == Approx(original_LeftCoord[14]));
	CHECK(Length[14] == Approx(original_Length[14]));
		
	CHECK(LeftCoord[16] == Approx(original_LeftCoord[16]));
	CHECK(Length[16] == Approx(original_Length[16]));	
}

TEST_CASE("Filtering with one inactive regions")
{
	constexpr int ndim = 2;
	Sub_regions<double, ndim> coordinates(10);
	size_t nregions = coordinates.size;
	Region_characteristics<ndim> classifications(nregions);
	Sub_regions_filter<double, ndim> filter(nregions);
	Region_estimates<double, ndim> estimates(nregions);
	Region_estimates<double, ndim> parents(nregions/2);

	auto integrals_mirror = Kokkos::create_mirror_view(estimates.integral_estimates);
	auto error_mirror = Kokkos::create_mirror_view(estimates.error_estimates);
	auto parent_integrals_mirror = Kokkos::create_mirror_view(parents.integral_estimates);
	auto parent_error_mirror = Kokkos::create_mirror_view(parents.error_estimates);	
	auto sub_dividing_dim = Kokkos::create_mirror_view(classifications.sub_dividing_dim);	
	auto host_active_regions = Kokkos::create_mirror_view(classifications.active_regions);	
	auto original_LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);	
	auto original_Length = Kokkos::create_mirror_view(coordinates.dLength);	
	
	Kokkos::deep_copy(original_LeftCoord, coordinates.dLeftCoord);
	Kokkos::deep_copy(original_Length, coordinates.dLength);
	
	for(int i=0; i < nregions; ++i){
		integrals_mirror[i] = 1000.;
		error_mirror[i] = 10.;
		host_active_regions[i] = 1;
		sub_dividing_dim[i] = 0; //active regions all have 0 as sub-dividing dim
			
		if(i < nregions/2){
			parent_integrals_mirror[i] = 1500.;
			parent_error_mirror[i] = 20.;
		}
	}
		
	//give values for inactive region(s)
	integrals_mirror[15] = 999.;
	error_mirror[15] = 1.;
	host_active_regions[15] = 0; 
	sub_dividing_dim[15] = 1; //active region has 1 as sub-dividing dim
		
	Kokkos::deep_copy(classifications.active_regions, host_active_regions);
	Kokkos::deep_copy(estimates.integral_estimates, integrals_mirror);
	Kokkos::deep_copy(estimates.error_estimates, error_mirror);
	Kokkos::deep_copy(parents.integral_estimates, parent_integrals_mirror);
	Kokkos::deep_copy(parents.error_estimates, parent_error_mirror);
	Kokkos::deep_copy(sub_dividing_dim, classifications.sub_dividing_dim);
	
	size_t num_active = filter.filter(coordinates, classifications, estimates, parents);
		
	//after filter, the pre-filtering integral and error-estimates become the parents
	auto filtered_integrals = Kokkos::create_mirror_view(parents.integral_estimates);
	auto filtered_errors = Kokkos::create_mirror_view(parents.error_estimates);
	
	auto new_subdiv_dim = Kokkos::create_mirror_view(classifications.sub_dividing_dim);	
	auto LeftCoord = Kokkos::create_mirror_view(coordinates.dLeftCoord);	
	auto Length = Kokkos::create_mirror_view(coordinates.dLength);	
	
	Kokkos::deep_copy (LeftCoord, coordinates.dLeftCoord);	
	Kokkos::deep_copy (Length, coordinates.dLength);	
	Kokkos::deep_copy (filtered_integrals, parents.integral_estimates);	
	Kokkos::deep_copy (filtered_errors, parents.error_estimates);	
	Kokkos::deep_copy (new_subdiv_dim, classifications.sub_dividing_dim);	
	
	//check index the index inactive region to see if filtered out, check surrounding indices to detect any errors 
	CHECK(filtered_integrals[16] == 1000.);
	CHECK(filtered_integrals[15] == 1000.);
	CHECK(filtered_integrals[14] == 1000.);
		
	CHECK(filtered_errors[16] == 10.);
	CHECK(filtered_errors[15] == 10.);
	CHECK(filtered_errors[14] == 10.);
		
	CHECK(new_subdiv_dim[16] == 0);
	CHECK(new_subdiv_dim[15] == 0);
	CHECK(new_subdiv_dim[14] == 0);
	
	//check regions 0, 1 for being the same and region 15 for being diffrent
	
	SECTION("Region 0, 1, 14 are the same")
	{
		for(int dim = 0; dim < ndim; ++dim){
			CHECK(Length[0 + dim * nregions] == Approx(original_Length[0 + dim * nregions]));
			CHECK(LeftCoord[0 + dim * nregions] == Approx(original_LeftCoord[0 + dim * nregions]));
			
			CHECK(Length[1 + dim * nregions] == Approx(original_Length[1 + dim * nregions]));
			CHECK(LeftCoord[1 + dim * nregions] == Approx(original_LeftCoord[1 + dim * nregions]));
			
			CHECK(Length[14 + dim * nregions] == Approx(original_Length[14 + dim * nregions]));
			CHECK(LeftCoord[14 + dim * nregions] == Approx(original_LeftCoord[14 + dim * nregions]));
		}
	}
	
	SECTION("Region 15 is different same")
	{
	
		CHECK(LeftCoord[14 + 0 * nregions] == Approx(original_LeftCoord[14 + 0 * nregions]));	
		CHECK(LeftCoord[14 + 1 * nregions] == Approx(original_LeftCoord[14 + 1 * nregions]));	

	}
}