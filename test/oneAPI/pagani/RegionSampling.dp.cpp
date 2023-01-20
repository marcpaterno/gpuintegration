#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include <iostream>
#include "oneAPI/pagani/quad/GPUquad/PaganiUtils.dp.hpp"
#include "common/oneAPI/cudaMemoryUtil.h"

class PTest {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    double res = 15.37;
    return res;
  }
};

class NTest {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    double res = -15.37;
    return res;
  }
};

class ZTest {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    return 0.;
  }
};

TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  int iteration = 0;
  bool compute_relerr_error_reduction = false;
  
  PTest integrand;
  PTest* d_integrand = quad::make_gpu_integrand(integrand);
  quad::Volume<double, ndim> vol;
  Cubature_rules<ndim> rules;
  rules.set_device_volume(vol.lows, vol.highs);
  double integral_val = 15.37;
  
  for(int splits_per_dim = 5; splits_per_dim < 15; splits_per_dim++){
	Sub_regions<ndim> sub_regions(splits_per_dim);
	size_t nregions = sub_regions.size;
	Region_characteristics<ndim> characteristics(nregions);
	Region_estimates<ndim> estimates(nregions);
	
	auto result = rules.template apply_cubature_integration_rules<PTest>(
			d_integrand,
			&sub_regions,
			&estimates,
			&characteristics,
			compute_relerr_error_reduction);
			
	double* h_estimates = quad::copy_to_host<double>(estimates.integral_estimates, nregions);
  
	double sum = 0.;
	for(size_t i = 0; i < nregions; ++i)
		sum += h_estimates[i];

	double true_val = integral_val/nregions;
	
	for(size_t i = 0; i < nregions; ++i){
		CHECK(h_estimates[i] == Approx(true_val).epsilon(1.e-6));
	}
  }	
}

TEST_CASE("Negative Positive Value Function")
{
  constexpr int ndim = 2;
  int iteration = 0;
  bool compute_relerr_error_reduction = false;
  
  NTest integrand;
  NTest* d_integrand = quad::make_gpu_integrand(integrand);
  quad::Volume<double, ndim> vol;
  Cubature_rules<ndim> rules;
  rules.set_device_volume(vol.lows, vol.highs);
  double integral_val = -15.37;
  
  for(int splits_per_dim = 5; splits_per_dim < 15; splits_per_dim++){
	Sub_regions<ndim> sub_regions(splits_per_dim);
	size_t nregions = sub_regions.size;
	Region_characteristics<ndim> characteristics(nregions);
	Region_estimates<ndim> estimates(nregions);
	
	auto result = rules.template apply_cubature_integration_rules<NTest>(
			d_integrand,
			&sub_regions,
			&estimates,
			&characteristics,
			compute_relerr_error_reduction);
			
	double* h_estimates = quad::copy_to_host<double>(estimates.integral_estimates, nregions);
  
	double sum = 0.;
	for(size_t i = 0; i < nregions; ++i)
		sum += h_estimates[i];

	double true_val = integral_val/nregions;
	
	for(size_t i = 0; i < nregions; ++i){
		CHECK(h_estimates[i] == Approx(true_val).epsilon(1.e-6));
	}
  }	
}

TEST_CASE("Zero Positive Value Function")
{
  constexpr int ndim = 2;
  int iteration = 0;
  bool compute_relerr_error_reduction = false;
  
  ZTest integrand;
  ZTest* d_integrand = quad::make_gpu_integrand(integrand);
  quad::Volume<double, ndim> vol;
  Cubature_rules<ndim> rules;
  rules.set_device_volume(vol.lows, vol.highs);
  double integral_val = 0.;
  
  for(int splits_per_dim = 5; splits_per_dim < 15; splits_per_dim++){
	Sub_regions<ndim> sub_regions(splits_per_dim);
	size_t nregions = sub_regions.size;
	Region_characteristics<ndim> characteristics(nregions);
	Region_estimates<ndim> estimates(nregions);
	
	auto result = rules.template apply_cubature_integration_rules<ZTest>(
			d_integrand,
			&sub_regions,
			&estimates,
			&characteristics,
			compute_relerr_error_reduction);
			
	double* h_estimates = quad::copy_to_host<double>(estimates.integral_estimates, nregions);
  
	double sum = 0.;
	for(size_t i = 0; i < nregions; ++i)
		sum += h_estimates[i];

	double true_val = integral_val/nregions;
	
	for(size_t i = 0; i < nregions; ++i){
		CHECK(h_estimates[i] == Approx(true_val).epsilon(1.e-6));
	}
  }	
}