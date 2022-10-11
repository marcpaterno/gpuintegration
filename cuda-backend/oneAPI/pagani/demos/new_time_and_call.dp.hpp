#ifndef ALTERNATIVE_TIME_AND_CALL_CUH
#define ALTERNATIVE_TIME_AND_CALL_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include "oneAPI/pagani/quad/GPUquad/PaganiUtils.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Workspace.dp.hpp"
#include "oneAPI/pagani/quad/util/cuhreResult.dp.hpp"
#include "oneAPI/pagani/quad/util/Volume.dp.hpp"
#include <iostream>
#include <iomanip>

template <typename F, int ndim>
void
call_cubature_rules(F integrand, quad::Volume<double, ndim>&  vol){
	
	F* d_integrand = make_gpu_integrand<F>(integrand);
	size_t partitions_per_axis = 2;   
	if(ndim < 5)
		partitions_per_axis = 4;
	else if(ndim <= 10)
		partitions_per_axis = 2;
	else
		partitions_per_axis = 1;
	partitions_per_axis = 8;
	Sub_regions<ndim> sub_regions(partitions_per_axis);
	size_t num_regions = sub_regions.size;
	std::cout<<"Initial regions:"<<sub_regions.size << std::endl;
    Region_characteristics<ndim> characteristics(sub_regions.size);
    Region_estimates<ndim> estimates(sub_regions.size);
	Cubature_rules<ndim> rules;
    rules.set_device_volume(vol.lows, vol.highs);
	int iteration = 0;
	bool compute_relerr_error_reduction = false;
	cuhreResult<double> iter = rules.template apply_cubature_integration_rules<F>(d_integrand, iteration, &sub_regions, &estimates, &characteristics, compute_relerr_error_reduction);
	
	double estimate = reduction<double>(estimates.integral_estimates, num_regions);
	double errorest = reduction<double>(estimates.error_estimates, num_regions);
	
	std::cout << std::setprecision(15) << std::scientific << iter.estimate << "," << iter.errorest << std::endl;
	dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
	sycl::free(d_integrand, q_ct1);
}	

/*
    we are not keeping track of nFinished regions
    id, ndim, true_val, epsrel, epsabs, estimate, errorest, nregions,
   nFinishedRegions, status, time
*/

template <typename F, int ndim, bool use_custom = false, int debug = 0>
bool
clean_time_and_call(std::string id,
                    F integrand,
                    double epsrel,
                    double true_value,
                    char const* algname,
                    std::ostream& outfile,
                    bool relerr_classification = true)
{

  auto print_custom = [=](bool use_custom_flag) {
    std::string to_print = use_custom_flag == true ? "custom" : "library";
    return to_print;
  };

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;

  double constexpr epsabs = 1.0e-40;
  bool good = false;
  Workspace<ndim, use_custom> workspace;
  quad::Volume<double, ndim> vol;

  
  for(int i=0; i < 2; i++){
	auto const t0 = std::chrono::high_resolution_clock::now();
	size_t partitions_per_axis = 2;   
	if(ndim < 5)
		partitions_per_axis = 4;
	else if(ndim <= 10)
		partitions_per_axis = 2;
	else
		partitions_per_axis = 1;
	Sub_regions<ndim> sub_regions(partitions_per_axis);

    constexpr bool collect_iters = false;
    constexpr bool collect_sub_regions = false;
    constexpr bool predict_split = false;
    cuhreResult<double> result =
      workspace.template integrate<F,
                                   predict_split,
                                   collect_iters,
                                   collect_sub_regions,
                                   debug>(
        integrand, sub_regions, epsrel, epsabs, vol, relerr_classification);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    double const absolute_error = std::abs(result.estimate - true_value);

    if (result.status == 0) {
      good = true;
    }

    outfile.precision(17);
    if (i != 0.)
      outfile << std::fixed << std::scientific << "pagani"
              << "," << id << "," << ndim << "," << print_custom(use_custom)
              << "," << true_value << "," << epsrel << "," << epsabs << ","
              << result.estimate << "," << result.errorest << ","
              << result.nregions << "," << result.nFinishedRegions << ","
              << result.status << "," << dt.count() << std::endl;
  }
  return good;
}

void
print_header()
{
  std::cout << "id, ndim, integral, epsrel, epsabs, estimate, errorest, "
               "nregions, status, time\n";
}

#endif
