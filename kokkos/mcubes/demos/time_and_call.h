#ifndef KOKKOS_MCUBES_DEMO_UTILS_CUH
#define KOKKOS_MCUBES_DEMO_UTILS_CUH

#include <iomanip>
#include <chrono>
#include <iostream>
#include <map>
#include <string>

#include "kokkos/mcubes/mcubes.h"
#include "common/kokkos/Volume.cuh"

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

struct VegasParams {

  VegasParams(double callsPerIter,
              double total_iters,
              double adjust_iters,
              int skipIters)
    : ncall(callsPerIter)
    , t_iter(total_iters)
    , num_adjust_iters(adjust_iters)
    , num_skip_iters(skipIters){};

  double ncall = 1.e7;
  int t_iter = 70;
  int num_adjust_iters = 40;
  int num_skip_iters = 5;
};

bool
CanAdjustNcallOrIters(double ncall, int totalIters)
{
  if (ncall >= 8.e9 && totalIters >= 100)
    return false;
  else
    return true;
}

bool
AdjustParams(double& ncall, int& totalIters)
{
  if (ncall >= 8.e9 && totalIters >= 100) {
    // printf("Adjusting will return false\n");
    return false;
  } else if (ncall >= 8.e9) {
    //  printf("Adjusting will increase iters by 10 current value:%i\n",
    //  totalIters);
    totalIters += 10;
    return true;
  } else if (ncall >= 1.e9) {
    // printf("Adjusting will increase ncall by 1e9 current value:%e\n", ncall);
    ncall += 1.e9;
    return true;
  } else {
    //  printf("Adjusting will multiply ncall by 10 current value:%e\n", ncall);
    ncall *= 10.;
    return true;
  }
}

void
PrintHeader()
{
  std::cout << "platform, alg, id, epsrel, integral, estimate, errorest, chi, "
               "iters, adj_iters, skip_iters, completed_iters, ncall,"
               "time, status\n";
}

template <typename F, int ndim, typename GeneratorType = kokkos_mcubes::Curand_generator>
bool
mcubes_time_and_call(F integrand,
                     double epsrel,
                     double correct_answer,
                     char const* id,
                     VegasParams& params,
                     quad::Volume<double, ndim> const* volume)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-20;
  bool success = false;
  int run = 0;

  do {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto res = kokkos_mcubes::integrate<F, ndim, GeneratorType>(integrand,
                                                 epsrel,
                                                 epsabs,
                                                 params.ncall,
                                                 volume,
                                                 params.t_iter,
                                                 params.num_adjust_iters,
                                                 params.num_skip_iters);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    success = (res.status == 0);
    std::cout.precision(15);

    if (success)
      std::cout << "kokkos, mcubes"
                << "," << id << "," << epsrel << "," << std::scientific
                << correct_answer << "," << std::scientific << res.estimate
                << "," << std::scientific << res.errorest << "," << res.chi_sq
                << "," << params.t_iter << "," << params.num_adjust_iters << ","
                << params.num_skip_iters << "," << res.iters << ","
                << params.ncall << "," << dt.count() << "," << res.status
                << "\n";
    else
      std::cout << "failed\n";
    // break;
    if (run == 0 && !success)
      AdjustParams(params.ncall, params.t_iter);
    if (success)
      run++;
  } while (success == false &&
           CanAdjustNcallOrIters(params.ncall, params.t_iter) == true);

  return success;
}


template <typename F, int ndim>
bool
signle_invocation_time_and_call(F integrand,
                                double epsrel,
                                double correct_answer,
                                char const* integralName,
                                VegasParams& params,
                                quad::Volume<double, ndim> const* volume,
                                int num_repeats = 100)
{

  bool success = false;
  for (int i = 0; i < num_repeats; ++i) {

    using MilliSeconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;
    // We make epsabs so small that epsrel is always the stopping condition.
    double constexpr epsabs = 1.0e-20;
	std::cout<<"about to call integrate"<<std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto res = kokkos_mcubes::integrate<F, ndim, kokkos_mcubes::Custom_generator>(
      integrand,
      epsrel,
      epsabs,
      params.ncall,
      volume,
      params.t_iter,
      params.num_adjust_iters,
      params.num_skip_iters);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    success = (res.status == 0);
    std::cout.precision(15);
    
    std::cout << integralName << "," << epsrel << "," << std::scientific
                << correct_answer << "," << std::scientific << res.estimate
                << "," << std::scientific << res.errorest << "," << res.chi_sq
                << "," << params.t_iter << "," << params.num_adjust_iters << ","
                << params.num_skip_iters << "," << res.iters << ","
                << params.ncall << "," << res.neval << "," << dt.count() << ","
                << res.status << "\n";
  }

  return success;
}

template <typename F, int ndim>
void
call_mcubes_kernel(int num_repeats)
{
  std::array<double, 4> required_ncall = {1.e8, 1.e9, 2.e9, 3.e9};
  double ncall = 1.0e8;
  int titer = 1;
  int itmax = 1;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);
  F integrand;
  quad::Volume<double, ndim> volume;
  size_t run = 0;
  double epsrel = 1.e-3;
  double true_value = 0.;
  
  for (auto num_samples : required_ncall) {
    params.ncall = num_samples;

    signle_invocation_time_and_call<F, ndim>(
      integrand, epsrel, true_value, "f", params, &volume, num_repeats);
    run++;
    if (run > required_ncall.size())
      break;
  }
}

#endif