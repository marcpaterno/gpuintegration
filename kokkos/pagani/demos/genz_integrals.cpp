#include "kokkos/pagani/demos/demo_utils.cuh"
#include "common/kokkos/integrands.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename F, int ndim, bool use_custom = false, int debug = 0, int num_runs = 10, bool collect_mult_runs = false>
bool
clean_time_and_call(std::string id, double epsrel, std::ostream& outfile)
{
  auto print_custom = [=](bool use_custom_flag) {
    std::string to_print = use_custom_flag == true ? "custom" : "library";
    return to_print;
  };

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;

  double constexpr epsabs = 1.0e-40;
  bool good = false;
  Workspace<double, ndim, use_custom, collect_mult_runs> workspace;
  F integrand;
  integrand.set_true_value();
  const quad::Volume<double, ndim> vol;
  size_t partitions_per_axis = get_partitions_per_axis(ndim);

  for (int i = 0; i < num_runs; i++) {
    Sub_regions<double, ndim> subregions(partitions_per_axis);
    auto const t0 = std::chrono::high_resolution_clock::now();

    constexpr bool collect_iters = false;
    constexpr bool predict_split = false;
    bool relerr_classification = true;
    numint::integration_result result =
      workspace.template integrate<F, predict_split, collect_iters, debug>(
        integrand, subregions, epsrel, epsabs, vol, relerr_classification, id);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  
    if (result.status == 0) {
      good = true;
    }

    std::cout.precision(17);
    if (i != 0.)
    std::cout << std::fixed << std::scientific << "pagani"
              << "," << id << "," << ndim << "," << print_custom(use_custom)
              << "," << integrand.true_value << "," << epsrel << "," << epsabs << ","
              << result.estimate << "," << result.errorest << ","
              << result.nregions << "," << result.nFinishedRegions << ","
              << result.iters  << ","
              << result.status << "," << dt.count() << std::endl;

    if (i != 0.)
      outfile << std::fixed << std::scientific << id << "," << ndim << ","
            << print_custom(use_custom) << "," << integrand.true_value << ","
            << epsrel << "," << epsabs << "," << result.estimate << ","
            << result.errorest << "," << result.nregions << "," 
            << result.nFinishedRegions << ","
            << result.iters << "," 
            << result.status << "," << dt.count() << std::endl;        
  }
  return good;
}

int
main()
{
  Kokkos::initialize();
  {
  std::vector<double> epsrels = {
    1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9};
  std::ofstream outfile("kokkos_pagani_genz_integrals_custom.csv");
  constexpr bool use_custom = true;
  constexpr int debug = 1;
  constexpr int num_runs = 3;
  constexpr bool collect_mult_runs = true;
  outfile << "id, ndim, use_custom, true_value,  ,epsabs, estimate, errorest, "
             "nregions, completed_iters, status, time"
          << std::endl;

  for (double epsrel : epsrels) {
    constexpr int ndim = 8;
    clean_time_and_call<F_1_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f1", epsrel, outfile);
    clean_time_and_call<F_2_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f2", epsrel, outfile);
    clean_time_and_call<F_3_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f3", epsrel, outfile);
    clean_time_and_call<F_4_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f4", epsrel, outfile);
    clean_time_and_call<F_5_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f5", epsrel, outfile);
    clean_time_and_call<F_6_8D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f6", epsrel, outfile);
  }

  for (double epsrel : epsrels) {
    constexpr int ndim = 7;
    clean_time_and_call<F_1_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f1", epsrel, outfile);
    clean_time_and_call<F_2_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f2", epsrel, outfile);
    clean_time_and_call<F_3_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f3", epsrel, outfile);
    clean_time_and_call<F_4_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f4", epsrel, outfile);
    clean_time_and_call<F_5_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f5", epsrel, outfile);
    clean_time_and_call<F_6_7D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f6", epsrel, outfile);
  }

  for (double epsrel : epsrels) {
    constexpr int ndim = 6;
    clean_time_and_call<F_1_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f1", epsrel, outfile);
    clean_time_and_call<F_2_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f2", epsrel, outfile);
    clean_time_and_call<F_3_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f3", epsrel, outfile);
    clean_time_and_call<F_4_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f4", epsrel, outfile);
    clean_time_and_call<F_5_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f5", epsrel, outfile);
    clean_time_and_call<F_6_6D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f6", epsrel, outfile);
  }

  for (double epsrel : epsrels) {
    constexpr int ndim = 5;
    clean_time_and_call<F_1_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f1", epsrel, outfile);
    clean_time_and_call<F_2_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f2", epsrel, outfile);
    clean_time_and_call<F_3_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f3", epsrel, outfile);
    clean_time_and_call<F_4_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f4", epsrel, outfile);
    clean_time_and_call<F_5_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f5", epsrel, outfile);
    clean_time_and_call<F_6_5D, ndim, use_custom, debug, num_runs, collect_mult_runs>(
      "f6", epsrel, outfile);
  }

  outfile.close();
  }
  Kokkos::finalize();
  return 0;
}