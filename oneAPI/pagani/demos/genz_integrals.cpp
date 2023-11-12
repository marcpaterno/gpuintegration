#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

size_t
get_partitions_per_axis(int ndim)
{
  size_t splits = 0;
  if (ndim < 5)
    splits = 4;
  else if (ndim <= 10)
    splits = 2;
  else
    splits = 1;
  return splits;
}

using namespace quad;

template <typename F, int ndim, bool use_custom = false, int debug = 0, int num_runs = 10, bool collect_mult_runs = false>
bool
new_clean_time_and_call(std::string id, double epsrel, std::ostream& outfile)
{

  auto print_custom = [=](bool use_custom_flag) {
    std::string to_print = use_custom_flag == true ? "custom" : "library";
    return to_print;
  };

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;

  double constexpr epsabs = 1.0e-40;
  bool good = false;
  Workspace<ndim, use_custom, collect_mult_runs> workspace;
  F integrand;

  //warpup execution
 
  integrand.set_true_value();
  quad::Volume<double, ndim> vol;
  size_t partitions_per_axis = get_partitions_per_axis(ndim);
  numint::integration_result result;

  for (int i = 0; i < num_runs; i++) {
    auto const t0 = std::chrono::high_resolution_clock::now();
    size_t partitions_per_axis = 2;
    if (ndim < 5)
      partitions_per_axis = 4;
    else if (ndim <= 10)
      partitions_per_axis = 2;
    else
      partitions_per_axis = 1;

    Sub_regions<ndim> sub_regions(partitions_per_axis);
    constexpr bool collect_iters = false;
    constexpr bool collect_sub_regions = false;
    constexpr bool predict_split = false;
    bool relerr_classification = true;

    /*if(i == 0){
      //warmp up execution
      result =
      workspace.template integrate<F,
                                   predict_split,
                                   collect_iters,
                                   collect_sub_regions,
                                   false>(
        integrand, sub_regions, epsrel, epsabs, vol, relerr_classification, id);
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;  

      std::cout << std::fixed << std::scientific << id << "," << ndim << "," << print_custom(use_custom)
              << "," << integrand.true_value << "," << epsrel << "," << epsabs << ","
              << result.estimate << "," << result.errorest << ","
              << result.nregions << "," << result.nFinishedRegions << ","
              << result.iters  << ","
              << result.status << "," << dt.count() << std::endl;
    }
    else*/
    {

      result = workspace.template integrate<F,
                                   predict_split,
                                   collect_iters,
                                   collect_sub_regions,
                                   debug>(
      integrand, sub_regions, epsrel, epsabs, vol, relerr_classification, id);
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
      std::cout.precision(17);
    
      std::cout << std::fixed << std::scientific << id << "," << ndim << "," << print_custom(use_custom)
              << "," << integrand.true_value << "," << epsrel << "," << epsabs << ","
              << result.estimate << "," << result.errorest << ","
              << result.nregions << "," << result.nFinishedRegions << ","
              << result.iters  << ","
              << result.status << "," << dt.count() << std::endl;

      outfile << std::fixed << std::scientific << id << "," << ndim << ","
            << print_custom(use_custom) << "," << integrand.true_value << ","
            << epsrel << "," << epsabs << "," << result.estimate << ","
            << result.errorest << "," << result.nregions << "," 
            << result.nFinishedRegions << ","
            << result.iters << "," 
            << result.status << "," << dt.count() << std::endl;        
    }
 

    if (result.status == 0) {
      good = true;
    }
  }
  return good;
}

int
main()
{
  ShowDevice(dpct::get_default_queue());
 std::vector<double> epsrels = {
    1.e-3/*, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9*/};
  std::ofstream outfile("oneapi_pagani_genz_integrals_custom.csv");
  constexpr bool use_custom = true;
  constexpr int debug = 1;
  constexpr int num_runs = 2;
  constexpr bool collect_mult_runs = true;
  outfile << "id, ndim, use_custom, true_value,  epsrel, epsabs, estimate, errorest, "
             "nregions, nfinished_regions, completed_iters, status, time"
          << std::endl;
  
  for (double epsrel : epsrels) {
    constexpr int ndim = 8;
    new_clean_time_and_call<F_1_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f1", epsrel, outfile);   
    new_clean_time_and_call<F_2_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f2", epsrel, outfile);
    /*new_clean_time_and_call<F_3_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>("f3", epsrel, outfile);
    new_clean_time_and_call<F_4_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f4", epsrel, outfile);
    new_clean_time_and_call<F_5_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f5", epsrel, outfile);
    new_clean_time_and_call<F_6_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f6", epsrel, outfile);*/
  
  }

  /*for (double epsrel : epsrels) {
    constexpr int ndim = 7;
    new_clean_time_and_call<F_1_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f1", epsrel, outfile);
    new_clean_time_and_call<F_2_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f2", epsrel, outfile);
    new_clean_time_and_call<F_3_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f3", epsrel, outfile);
    new_clean_time_and_call<F_4_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f4", epsrel, outfile);
    new_clean_time_and_call<F_5_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f5", epsrel, outfile);
    new_clean_time_and_call<F_6_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f6", epsrel, outfile);
  }

  for (double epsrel : epsrels) {
    constexpr int ndim = 6;
    new_clean_time_and_call<F_1_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f1", epsrel, outfile);
    new_clean_time_and_call<F_2_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f2", epsrel, outfile);
    new_clean_time_and_call<F_3_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f3", epsrel, outfile);
    new_clean_time_and_call<F_4_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f4", epsrel, outfile);
    new_clean_time_and_call<F_5_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f5", epsrel, outfile);
    new_clean_time_and_call<F_6_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f6", epsrel, outfile);
  }

  for (double epsrel : epsrels) {
    constexpr int ndim = 5;
    new_clean_time_and_call<F_1_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f1", epsrel, outfile);
    new_clean_time_and_call<F_2_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f2", epsrel, outfile);
    new_clean_time_and_call<F_3_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f3", epsrel, outfile);
    new_clean_time_and_call<F_4_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f4", epsrel, outfile);
    new_clean_time_and_call<F_5_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f5", epsrel, outfile);
    new_clean_time_and_call<F_6_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f6", epsrel, outfile);
  }*/

  outfile.close();

}
