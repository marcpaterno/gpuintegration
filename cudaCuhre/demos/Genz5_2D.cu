#include "function.cuh"
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

template <typename F>
bool
time_and_call(std::string id,
              F integrand,
              double epsrel,
              double true_value,
              char const* algname,
              std::ostream& outfile,
              int _final = 0)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-40;

  double lows[] = {0., 0};
  double highs[] = {1., 1.};

  constexpr int ndim = 2;
  quad::Volume<double, ndim> vol(lows, highs);
  int const key = 0;
  int const verbose = 0;
  int const numdevices = 1;
  quad::Cuhre<double, ndim> alg(0, nullptr, key, verbose, numdevices);

  int outfileVerbosity = 0;
  constexpr int phase_I_type = 0; // alternative phase 1

  auto const t0 = std::chrono::high_resolution_clock::now();
  cuhreResult const result = alg.integrate<F>(
    integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  bool good = false;

  if (result.status == 0 || result.status == 2) {
    good = true;
  }

  outfile << std::fixed << id << ",\t" << std::fixed << true_value << ",\t"
          << std::scientific << epsrel << ",\t\t\t" << std::scientific
          << epsabs << ",\t" << std::fixed << result.estimate << ",\t"
          << std::fixed << result.errorest << ",\t" << std::fixed
          << result.nregions << ",\t" << std::fixed << result.status << ",\t"
          << _final << ",\t" << result.lastPhase << ",\t" << dt.count() << std::endl;
  return good;
}

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double const epsrel_min = 1.0e-12;
  double true_value = 0.039462780237263662026;
  GENZ_5_2D integrand;
  std::cout << "id, value, epsrel, epsabs, estimate, errorest, regions, "
             "converge, final, total_time\n";
  int _final = 1;
  while (time_and_call("pdc_f1_latest",
                       integrand,
                       epsrel,
                       true_value,
                       "gpucuhre",
                       std::cout,
                       _final) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  _final = 0;
  epsrel = 1.0e-3;

  while (time_and_call("pdc_f0_latest",
                       integrand,
                       epsrel,
                       true_value,
                       "gpucuhre",
                       std::cout,
                       _final) == true &&
         epsrel >= epsrel_min) {
    epsrel = epsrel >= 1e-6 ? epsrel / 5.0 : epsrel / 2.0;
  }
}
