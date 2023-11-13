#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "common/cuda/integrands.cuh"

template <typename F,
          int ndim,
          bool use_custom = false,
          int debug = 0,
          int runs_per_esprel = 10>
bool
g_func_time_and_call(std::string id,
                     double epsrel,
                     std::ostream& outfile,
                     quad::Volume<double, ndim>& vol)
{

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-40;
  bool good = false;
  bool relerr_classification = true;
  Workspace<double, ndim, debug, use_custom> workspace;
  F integrand;
  integrand.set_true_value(vol.lows[0], vol.highs[0]);
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
      workspace.template integrate<F, predict_split, collect_iters>(
        integrand, sub_regions, epsrel, epsabs, vol, relerr_classification);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    double const absolute_error =
      std::abs(result.estimate - integrand.true_value);

    if (result.status == 0) {
      good = true;
    }

    std::cout.precision(17);
    /*if (i != 0)*/ {
      std::cout << id << "," << ndim << "," << vol.lows[0] << ","
                << vol.highs[0] << "," << print_custom(use_custom) << ","
                << std::fixed << std::scientific << integrand.true_value << ","
                << epsrel << "," << epsabs << "," << result.estimate << ","
                << result.errorest << "," << result.nregions << ","
                << result.status << "," << dt.count() << std::endl;

      outfile << id << "," << ndim << "," << std::scientific << vol.lows[0]
              << "," << vol.highs[0] << "," << print_custom(use_custom) << ","
              << integrand.true_value << "," << epsrel << "," << epsabs << ","
              << result.estimate << "," << result.errorest << ","
              << result.nregions << "," << result.iters << "," << result.status
              << "," << dt.count() << std::endl;
    }
  }
  return good;
}

int
main()
{
  constexpr bool use_custom = false;
  constexpr int debug = 0;
  constexpr int num_runs = 10;
  std::vector<double> epsrels = {
    1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9};
  std::vector<std::pair<double, double>> volumes = {{0, 1}, {0, 2}, {0, 3}};
  std::ofstream outfile("cuda_pagani_g_func.csv");
  outfile << "id, ndim, low, high, use_custom, true_value, epsrel, epsabs, "
             "estimate, errorest, nregions, completed_iters, status, time"
          << std::endl;

  for (auto volume : volumes) {

    for (double epsrel : epsrels) {
      constexpr int ndim = 10;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      g_func_time_and_call<G_func_10D, ndim, use_custom, debug, num_runs>(
        "G_func", epsrel, outfile, vol);
    }

    for (double epsrel : epsrels) {
      constexpr int ndim = 9;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      g_func_time_and_call<G_func_9D, ndim, use_custom, debug, num_runs>(
        "G_func", epsrel, outfile, vol);
    }

    for (double epsrel : epsrels) {
      constexpr int ndim = 8;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      g_func_time_and_call<G_func_8D, ndim, use_custom, debug, num_runs>(
        "G_func", epsrel, outfile, vol);
    }

    for (double epsrel : epsrels) {
      constexpr int ndim = 7;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      g_func_time_and_call<G_func_7D, ndim, use_custom, debug, num_runs>(
        "G_func", epsrel, outfile, vol);
    }

    for (double epsrel : epsrels) {
      constexpr int ndim = 6;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      g_func_time_and_call<G_func_6D, ndim, use_custom, debug, num_runs>(
        "G_func", epsrel, outfile, vol);
    }

    for (double epsrel : epsrels) {
      constexpr int ndim = 5;
      quad::Volume<double, ndim> vol(volume.first, volume.second);
      g_func_time_and_call<G_func_5D, ndim, use_custom, debug, num_runs>(
        "G_func", epsrel, outfile, vol);
    }
  }

  outfile.close();
  return 0;
}
