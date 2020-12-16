#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

template <typename ALG, typename F>
bool
time_and_call(ALG const& a, F f, double epsrel, double correct_answer, char const* algname)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-16;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = a.integrate(f, epsrel, epsabs);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double absolute_error = std::abs(res.value - correct_answer);
  bool const good = (res.status == 0);
  std::cout << std::scientific
           << algname << '\t'
            << epsrel << '\t';
  if (good) {
    std::cout << res.value << '\t'
              << res.error << '\t'
              << absolute_error << '\t';
  } else {
    std:: cout << "NA\tNA\tNA\t";
  }
  std::cout << res.neval << '\t'
            << res.nregions << '\t'
            << dt.count()
            << std::endl;
  return res.status == 0;
}

struct Config{
    Config(int verbosity, int heuristic, int phaseT , int deviceNum, int finFlag): 
        outfileVerbosity{verbosity}, phase_I_type(phaseT), numdevices{deviceNum}, heuristicID{heuristic}, _final(finFlag){}
    Config() = default;
    
    int phase_I_type = 0;
    int outfileVerbosity = 0;
    int numdevices = 1;
    int heuristicID = 0;
    int _final = 1;
    int verbose = 0;
};

template <typename F, int ndim>
bool
cu_time_and_call(std::string id,
              F integrand,
              double epsrel,
              double true_value,
              char const* algname,
              std::ostream& outfile,
              Config config = Config(), 
              quad::Volume<double, ndim>* vol = nullptr)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-22;
  
  quad::Cuhre<double, ndim> alg(0, nullptr);
  
  auto const t0 = std::chrono::high_resolution_clock::now();
  cuhreResult const result = alg.integrate(
    integrand, epsrel, epsabs, vol, config.outfileVerbosity, config._final, config.heuristicID, config.phase_I_type);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  bool good = false;

  if (result.status == 0 || result.status == 2) {
    good = true;
  }
  
 outfile.precision(17);
 outfile << std::fixed  << std::scientific 
          << id << ",\t" << true_value << ",\t"
          << epsrel << ",\t" << epsabs << ",\t" << result.estimate << ",\t"
          << result.errorest << ",\t" << result.nregions << ",\t" << result.nFinishedRegions 
          << ",\t" << result.status << ",\t" << config._final << ",\t" 
          << result.lastPhase << ",\t" << dt.count() << std::endl;
  return good;
}