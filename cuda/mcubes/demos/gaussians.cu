#include <iostream>
#include "cuda/mcubes/demos/demo_utils.cuh"
#include "common/cuda/integrands.cuh"
#include "cuda/pagani/demos/compute_genz_integrals.cuh"

template <typename F, int ndim, int num_runs = 100>
void
gaussian_mcubes_time_and_call(std::string integ_id,
                              double epsrel,
                              VegasParams params,
                              std::ostream& outfile,
                              quad::Volume<double, ndim>& vol)
{
  std::vector<double> sharpness_params = {10, 15., 20., 25., 30.};
  for (auto sharpness : sharpness_params) {
    using MilliSeconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;
    // We make epsabs so small that epsrel is always the stopping condition.
    double constexpr epsabs = 1.0e-20;
    bool constexpr MCUBES_DEBUG = false;
    F integrand;
    integrand.sharpness = sharpness;
    integrand.peak_loc = .5;
    integrand.set_true_value();
    int run = 0;

    do {

      auto t0 = std::chrono::high_resolution_clock::now();
      auto res =
        cuda_mcubes::integrate<F, ndim, MCUBES_DEBUG>(integrand,
                                                      epsrel,
                                                      epsabs,
                                                      params.ncall,
                                                      &vol,
                                                      params.t_iter,
                                                      params.num_adjust_iters,
                                                      params.num_skip_iters);
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
      std::cout.precision(17);

      std::cout << integ_id << "," << std::scientific << ndim << ","
                << vol.lows[0] << "," << vol.highs[0] << ","
                << integrand.sharpness << "," << std::setprecision(15)
                << integrand.true_value << "," << epsrel << "," << epsabs << ","
                << std::scientific << res.estimate << "," << std::scientific
                << res.errorest << "," << res.chi_sq << "," << params.ncall
                << "," << params.t_iter << "," << params.num_adjust_iters << ","
                << res.iters << "," << dt.count() << "," << res.status << "\n";

      outfile << integ_id << "," << std::scientific << ndim << ","
              << vol.lows[0] << "," << vol.highs[0] << ","
              << integrand.sharpness << "," << std::setprecision(15)
              << integrand.true_value << "," << epsrel << "," << epsabs << ","
              << std::scientific << res.estimate << "," << std::scientific
              << res.errorest << "," << res.chi_sq << "," << params.ncall << ","
              << params.t_iter << "," << params.num_adjust_iters << ","
              << res.iters << "," << dt.count() << "," << res.status << "\n";
      run++;
      if (res.status == 1)
        break;
    } while (run < num_runs);
  }
}

int
main()
{
  std::ofstream outfile("cuda_mcubes_gaussians.csv");
  int titer = 300;
  int itmax = 20; // don't forget to adjust when comparing
  int skip = 5;   // that may need to be set to itmax
  constexpr int num_runs = 10;
  std::vector<std::pair<double, double>> volumes = {{0, 1}};
  std::vector<double> epsrels = {
    1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9};
  std::vector<double> required_ncall = {
    1.e5, 1.e6, 1.e7, 1.e8, 1.e9, 2.e9, 3.e9, 5.e9};
  VegasParams params(required_ncall[0], titer, itmax, skip);
  outfile << "id, ndim, low, high, sharpness, true_value, epsrel, epsabs, "
             "estimate, errorest, chi_sq, ncall, max_iters, adjust_iters, "
             "completed_iters, time, status"
          << std::endl;

  for (auto volume : volumes) {

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 8;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        gaussian_mcubes_time_and_call<F_4_8D_alt, ndim, num_runs>(
          "F_4_alt", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 7;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        gaussian_mcubes_time_and_call<F_4_7D_alt, ndim, num_runs>(
          "F_4_alt", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 6;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        gaussian_mcubes_time_and_call<F_4_6D_alt, ndim, num_runs>(
          "F_4_alt", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 5;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        gaussian_mcubes_time_and_call<F_4_5D_alt, ndim, num_runs>(
          "F_6_alt", epsrel, params, outfile, vol);
      }
    }
  }

  outfile.close();
  return 0;
}
