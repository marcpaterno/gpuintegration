#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "common/cuda/integrands.cuh"

template <typename F,
          int ndim,
          bool use_custom = false,
          int debug = 0,
          int runs_per_esprel = 10>
void
product_peak_time_and_call(std::string id,
                           double epsrel,
                           std::ostream& outfile,
                           quad::Volume<double, ndim>& vol)
{

  std::vector<double> peak_prominence = {40., 45., 50., 55., 60., 65., 75.};

  for (auto difficulty : peak_prominence) {
    using MilliSeconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;
    double constexpr epsabs = 1.0e-40;
    bool relerr_classification = true;
    Workspace<double, ndim, use_custom> workspace;
    F integrand;
    integrand.alpha = difficulty;
    integrand.set_true_value();
    auto print_custom = [=](bool use_custom_flag) {
      std::string to_print = use_custom_flag == true ? "custom" : "library";
      return to_print;
    };

    for (int i = 0; i < runs_per_esprel; i++) {
      auto const t0 = std::chrono::high_resolution_clock::now();
      size_t partitions_per_axis = get_partitions_per_axis(ndim);
      Sub_regions<double, ndim> sub_regions(partitions_per_axis);
      constexpr bool predict_split = false;
      constexpr bool collect_iters = false;

      numint::integration_result result =
        workspace.template integrate<F, predict_split, collect_iters, debug>(
          integrand, sub_regions, epsrel, epsabs, vol, relerr_classification);
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
      double const absolute_error =
        std::abs(result.estimate - integrand.true_value);

      std::cout.precision(17);
      /*if (i != 0)*/ {
        std::cout << id << "," << ndim << "," << vol.lows[0] << ","
                  << vol.highs[0] << "," << print_custom(use_custom) << ","
                  << difficulty << "," << std::fixed << std::scientific
                  << integrand.true_value << "," << epsrel << "," << epsabs
                  << "," << result.estimate << "," << result.errorest << ","
                  << result.nregions << "," << result.status << ","
                  << dt.count() << std::endl;

        outfile << id << "," << ndim << "," << std::scientific << vol.lows[0]
                << "," << vol.highs[0] << "," << print_custom(use_custom) << ","
                << difficulty << "," << integrand.true_value << "," << epsrel
                << "," << epsabs << "," << result.estimate << ","
                << result.errorest << "," << result.nregions << ","
                << result.iters << "," << result.status << "," << dt.count()
                << std::endl;
      }
    }
  }
}

int
main()
{
  constexpr bool use_custom = false;
  constexpr int debug = 0;
  constexpr int num_runs = 10;
  std::vector<double> epsrels = {
    1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9};
  std::vector<std::pair<double, double>> volumes = {{0, 1}};
  std::ofstream outfile("cuda_pagani_product_peaks.csv");
  outfile << "id, ndim, low, high, use_custom, peak_prominence, true_value, "
             "epsrel, epsabs, "
             "estimate, errorest, nregions, completed_iters, status, time"
          << std::endl;
  for (auto volume : volumes) {

    for (double epsrel : epsrels) {
      constexpr int ndim = 8;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      product_peak_time_and_call<F_2_8D_alt, ndim, use_custom, debug, num_runs>(
        "F_2_alt", epsrel, outfile, vol);
    }

    for (double epsrel : epsrels) {
      constexpr int ndim = 7;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      product_peak_time_and_call<F_2_7D_alt, ndim, use_custom, debug, num_runs>(
        "F_2_alt", epsrel, outfile, vol);
    }

    for (double epsrel : epsrels) {
      constexpr int ndim = 6;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      product_peak_time_and_call<F_2_6D_alt, ndim, use_custom, debug, num_runs>(
        "F_2_alt", epsrel, outfile, vol);
    }

    for (double epsrel : epsrels) {
      constexpr int ndim = 5;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      product_peak_time_and_call<F_2_5D_alt, ndim, use_custom, debug, num_runs>(
        "F_2_alt", epsrel, outfile, vol);
    }
  }

  outfile.close();
  return 0;
}
