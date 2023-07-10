#include <iostream>
#include "cuda/mcubes/demos/demo_utils.cuh"
#include "common/cuda/integrands.cuh"
#include "cuda/pagani/demos/compute_genz_integrals.cuh"
double constexpr PI = 3.14159265358979323844;

template <typename F, int ndim, int num_runs = 100>
bool
oscillatory_mcubes_time_and_call(std::string integ_id,
                                 double epsrel,
                                 VegasParams params,
                                 std::ostream& outfile,
                                 quad::Volume<double, ndim>& vol)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-20;
  bool constexpr MCUBES_DEBUG = false;
  bool success = false;
  F integrand;
  integrand.set_true_value(vol.lows[0], vol.highs[0]);
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
    success = (res.status == 0);
    std::cout.precision(17);

    std::cout << integ_id << "," << std::scientific << ndim << ","
              << vol.lows[0] << "," << vol.highs[0] << ","
              << std::setprecision(15) << integrand.true_value << "," << epsrel
              << "," << epsabs << "," << std::scientific << res.estimate << ","
              << std::scientific << res.errorest << "," << res.chi_sq << ","
              << params.ncall << "," << params.t_iter << ","
              << params.num_adjust_iters << "," << res.iters << ","
              << dt.count() << "," << res.status << "\n";

    outfile << integ_id << "," << std::scientific << ndim << "," << vol.lows[0]
            << "," << vol.highs[0] << "," << std::setprecision(15)
            << integrand.true_value << "," << epsrel << "," << epsabs << ","
            << std::scientific << res.estimate << "," << std::scientific
            << res.errorest << "," << res.chi_sq << "," << params.ncall << ","
            << params.t_iter << "," << params.num_adjust_iters << ","
            << res.iters << "," << dt.count() << "," << res.status << "\n";
    run++;
  } while (run < num_runs && success);

  return success;
}

int
main()
{
  std::ofstream outfile("cuda_mcubes_oscillatory_temp.csv");
  int titer = 300;
  int itmax = 0; // don't forget to adjust when comparing
  int skip = 0;  // that may need to be set to itmax
  constexpr int num_runs = 1;
  std::vector<std::pair<double, double>> volumes = {{0, 1.1*PI}/*, {0, 2.1* PI}, {0, 1.1* PI}, {0., 2.1 * PI}, {0., 3.1 * PI}, {0., 4.1 * PI}*/};
  std::vector<double> epsrels = {1.e-3 /*, 1.e-4, 1.e-5, 1.e-6*/};
  std::vector<double> required_ncall = {
    /*1.e5, 1.e6, 1.e7, 1.e8, 1.e9, 2.e9, 3.e9,*/ 5.e9};
  VegasParams params(required_ncall[0], titer, itmax, skip);

  outfile
    << "id, ndim, low, high, true_value, epsrel, epsabs, estimate, errorest, "
       "chi_sq, ncall, max_iters, adjust_iters, completed_iters, time, status"
    << std::endl;

  for (auto volume : volumes) {

    /*for(double ncall : required_ncall){
            params.ncall = ncall;
            for(double epsrel : epsrels){
                    constexpr int ndim = 1;
                    quad::Volume<double, ndim> vol(volume.first, volume.second);
                    oscillatory_mcubes_time_and_call<Cos_fully_sep_product_1D,
    ndim, num_runs>("fully_sep_osc",  epsrel, params, outfile, vol);
            }
    }*/

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 10;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        // oscillatory_mcubes_time_and_call<Oscillatory_10D, ndim,
        // num_runs>("non_sep_osc", epsrel, params, outfile, vol);
        oscillatory_mcubes_time_and_call<Cos_fully_sep_product_10D,
                                         ndim,
                                         num_runs>(
          "fully_sep_osc", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 9;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        // oscillatory_mcubes_time_and_call<Oscillatory_9D, ndim,
        // num_runs>("non_sep_osc", epsrel, params, outfile, vol);
        oscillatory_mcubes_time_and_call<Cos_fully_sep_product_9D,
                                         ndim,
                                         num_runs>(
          "fully_sep_osc", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 8;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        // oscillatory_mcubes_time_and_call<Oscillatory_8D, ndim,
        // num_runs>("non_sep_osc", epsrel, params, outfile, vol);
        // oscillatory_mcubes_time_and_call<Cos_semi_sep_product_8D, ndim,
        // num_runs>("semi_sep_osc",  epsrel, params, outfile, vol);
        oscillatory_mcubes_time_and_call<Cos_fully_sep_product_8D,
                                         ndim,
                                         num_runs>(
          "fully_sep_osc", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 7;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        // oscillatory_mcubes_time_and_call<Oscillatory_7D, ndim,
        // num_runs>("non_sep_osc", epsrel, params, outfile, vol);
        oscillatory_mcubes_time_and_call<Cos_fully_sep_product_7D,
                                         ndim,
                                         num_runs>(
          "fully_sep_osc", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 6;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        // oscillatory_mcubes_time_and_call<Oscillatory_6D, ndim,
        // num_runs>("non_sep_osc", epsrel, params, outfile, vol);
        // oscillatory_mcubes_time_and_call<Cos_semi_sep_product_6D, ndim,
        // num_runs>("semi_sep_osc",  epsrel, params, outfile, vol);
        oscillatory_mcubes_time_and_call<Cos_fully_sep_product_6D,
                                         ndim,
                                         num_runs>(
          "fully_sep_osc", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 5;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        // time_and_call_no_adjust_params<F_1_5D, ndim, num_runs>("F_1_5D",
        // epsrel, params, outfile, vol);
        // oscillatory_mcubes_time_and_call<Oscillatory_5D, ndim,
        // num_runs>("non_sep_osc", epsrel, params, outfile, vol);
        oscillatory_mcubes_time_and_call<Cos_fully_sep_product_5D,
                                         ndim,
                                         num_runs>(
          "fully_sep_osc", epsrel, params, outfile, vol);
      }
    }

    for (double ncall : required_ncall) {
      params.ncall = ncall;
      for (double epsrel : epsrels) {
        constexpr int ndim = 4;
        quad::Volume<double, ndim> vol(volume.first, volume.second);
        // oscillatory_mcubes_time_and_call<Oscillatory_4D, ndim,
        // num_runs>("non_sep_osc", epsrel, params, outfile, vol);
        // oscillatory_mcubes_time_and_call<Cos_semi_sep_product_4D, ndim,
        // num_runs>("semi_sep_osc", epsrel, params, outfile, vol);
        oscillatory_mcubes_time_and_call<Cos_fully_sep_product_4D,
                                         ndim,
                                         num_runs>(
          "fully_sep_osc", epsrel, params, outfile, vol);
      }
    }
  }

  outfile.close();
  return 0;
}
