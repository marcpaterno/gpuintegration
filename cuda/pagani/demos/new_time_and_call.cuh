#ifndef ALTERNATIVE_TIME_AND_CALL_CUH
#define ALTERNATIVE_TIME_AND_CALL_CUH

#include <chrono>
#include "cuda/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"
#include "common/integration_result.hh"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "nvToolsExt.h"
#include <string>
#include <iostream>
#include <iomanip>

template<typename T>
void host_print_dev_array(T* dev, size_t size, std::string label){
	T* host = new T[size];
	cuda_memcpy_to_host(host, dev, size);
	for(int i = 0; i < size; ++i)
		std::cout<<label << "," <<  i << ","  << std::scientific << std::setprecision(15) << host[i] << std::endl;
	delete[] host;
}

template <typename F, int ndim>
void
call_cubature_rules(F integrand, quad::Volume<double, ndim>& vol)
{
  // cudaDeviceReset();
  for(int splits_per_dim = 4; splits_per_dim < 10; splits_per_dim++){
	  F* d_integrand = make_gpu_integrand<F>(integrand);
	  Sub_regions<double, ndim> sub_regions(splits_per_dim);
	  size_t num_regions = sub_regions.size;
	  
	  if(num_regions >= 43e6)
		  break;
	  Region_characteristics<ndim> characteristics(sub_regions.size);
	  Region_estimates<double, ndim> estimates(sub_regions.size);
	  Cubature_rules<double, ndim> rules;
	  std::cout << "Initial regions:" << sub_regions.size << std::endl;
	  rules.set_device_volume(vol.lows, vol.highs);
	  int iteration = 0;
	  bool compute_relerr_error_reduction = false;
	  
	  numint::integration_result iter = rules.template apply_cubature_integration_rules<F>(
		d_integrand,
		iteration,
		sub_regions,
		estimates,
		characteristics,
		compute_relerr_error_reduction);
		
	  //sub_regions.print_bounds();
	  std::cout << "estimates:" << std::scientific << std::setprecision(15) << std::scientific << iter.estimate << "," << num_regions << std::endl;
	  cudaFree(d_integrand);
  }
}

template <typename F,
          typename T,
          int ndim,
          bool use_custom = false,
          int debug = 0>
bool
clean_time_and_call(std::string id,
                    F integrand,
                    T epsrel,
                    T true_value,
                    char const* algname,
                    std::ostream& outfile,
                    quad::Volume<T, ndim>& vol,
                    bool relerr_classification = true)
{

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  T constexpr epsabs = 1.0e-40;
  bool good = false;
  Workspace<T, ndim, use_custom> workspace;

  auto print_custom = [=](bool use_custom_flag) {
    std::string to_print = use_custom_flag == true ? "custom" : "library";
    return to_print;
  };

  for (int i = 0; i < 3; i++) {

    auto const t0 = std::chrono::high_resolution_clock::now();
    size_t partitions_per_axis = 2;
    if (ndim < 5)
      partitions_per_axis = 4;
    else if (ndim <= 10)
      partitions_per_axis = 2;
    else
      partitions_per_axis = 1;

    Sub_regions<T, ndim> sub_regions(partitions_per_axis);
    constexpr bool predict_split = false;
    constexpr bool collect_iters = false;

    numint::integration_result result =
      workspace.template integrate<F, predict_split, collect_iters, debug>(
        integrand, sub_regions, epsrel, epsabs, vol, relerr_classification);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    T const absolute_error = std::abs(result.estimate - true_value);

    if (result.status == 0) {
      good = true;
    }

    outfile.precision(17);
    if (i != 0)
      outfile << std::fixed << std::scientific << id << "," << ndim << ","
              << print_custom(use_custom) << "," << true_value << "," << epsrel
              << "," << epsabs << "," << result.estimate << ","
              << result.errorest << "," << result.nregions << ","
              << result.status << "," << dt.count() << std::endl;
  }
  return good;
}

void
print_header()
{
  std::cout << "id, ndim, use_custom, integral, epsrel, epsabs, estimate, "
               "errorest, nregions, status, time\n";
}
#endif
