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
  double constexpr epsabs = 1.0e-20;

  double lows[] =  {0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1.};

  constexpr int ndim = 5;
  quad::Volume<double, ndim> vol(lows, highs);
  int const key = 0;
  int const verbose = 0;
  int const numdevices = 1;
  quad::Cuhre<double, ndim> alg(0, nullptr, key, verbose, numdevices);

  int outfileVerbosity = 1;
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
 outfile.precision(17);
  outfile << std::fixed << id << ",\t" << std::scientific << true_value << ",\t"
          << epsrel << ",\t" << epsabs << ",\t" << result.estimate << ",\t"
          << result.errorest << ",\t" << result.nregions << ",\t" << result.nFinishedRegions 
          << ",\t" << result.status << ",\t" << _final << ",\t" 
          << result.lastPhase << ",\t" << dt.count() << std::endl;
  return good;
}

int
main()
{
  double epsrel =  1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 1.79132603674879e-06;
  GENZ_4_5D integrand;
  std::cout << "id, value, epsrel, epsabs, estimate, errorest, regions, fregions,"
             "converge, final, total_time\n";
  int _final = 1;
  while (time_and_call("Genz4_5D",
                       integrand,
                       epsrel,
                       true_value,
                       "gpucuhre",
                       std::cout,
                       _final) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  /*_final = 0;
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
  }*/
}
