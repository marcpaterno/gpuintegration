#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "cuda/pagani/quad/GPUquad/Pagani.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
#include "nvToolsExt.h"

//#ifdef USE_NVTX

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

template <typename ALG, typename F>
bool
time_and_call(ALG const& a,
              F f,
              double epsrel,
              double correct_answer,
              char const* algname)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-20;
  auto t0 = std::chrono::high_resolution_clock::now();

  auto res = a.integrate(f, epsrel, epsabs);

  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double absolute_error = std::abs(res.value - correct_answer);
  bool const good = (res.status == 0);
  std::cout << std::scientific << algname << '\t' << epsrel << '\t';
  if (good) {
    std::cout << res.value << '\t' << res.error << '\t' << absolute_error
              << '\t';
  } else {
    std::cout << "NA\tNA\tNA\t";
  }
  std::cout << res.neval << '\t' << res.nregions << '\t' << dt.count()
            << std::endl;
  return res.status == 0;
}

struct Config {
  Config(int verbosity,
         int heuristic,
         int phaseT,
         int deviceNum,
         int finFlag)
    : outfileVerbosity{verbosity}
    , phase_I_type(phaseT)
    , numdevices{deviceNum}
    , heuristicID{heuristic}
    , _final(finFlag)
  {}
  Config() = default;

  int phase_I_type = 0;
  bool phase_2 = false;
  int outfileVerbosity = 0;
  int numdevices = 1;
  int heuristicID = 0;
  int _final = 1;
  int verbose = 0;
};

void
PrintHeader()
{
  std::cout
    << "id, heuristicID, value, epsrel, epsabs, estimate, errorest, regions, finished_regions, "
       "status, final, lastPhase, total_time\n";
}

namespace floatIntegrands {

  template <typename F, int ndim>
  bool
  cu_time_and_call(std::string id,
                   F integrand,
                   float epsrel,
                   float true_value,
                   char const* algname,
                   std::ostream& outfile,
                   Config config = Config(),
                   quad::Volume<float, ndim>* vol = nullptr)
  {
    using MilliSeconds =
      std::chrono::duration<float, std::chrono::milliseconds::period>;
    float constexpr epsabs = 1.0e-20;

    quad::Pagani<float, ndim> alg;

    auto const t0 = std::chrono::high_resolution_clock::now();
    // nvtxRangePushA("init_host_data");
    cuhreResult const result = alg.integrate(integrand,
                                             epsrel,
                                             epsabs,
                                             vol,
                                             config.outfileVerbosity,
                                             config._final,
                                             config.heuristicID,
                                             config.phase_I_type);
    // nvtxRangePop();
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    float const absolute_error = std::abs(result.estimate - true_value);
    bool good = false;

    if (result.status == 0 || result.status == 2) {
      good = true;
    }

    outfile.precision(17);
    outfile << std::fixed << std::scientific << id << "," << config.heuristicID
            << "," << true_value << "," << epsrel << "," << epsabs << ","
            << result.estimate << "," << result.errorest << ","
            << result.nregions << "," << result.nFinishedRegions << ","
            << result.status << "," << config._final << "," << result.lastPhase
            << "," << dt.count() << std::endl;
    return good;
  }
}

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
  double constexpr epsabs = 1.0e-20;

  quad::Pagani<double, ndim> alg;

  auto const t0 = std::chrono::high_resolution_clock::now();
  // nvtxRangePushA("init_host_data");
  cuhreResult const result = alg.integrate(integrand,
                                           epsrel,
                                           epsabs,
                                           vol,
                                           config.outfileVerbosity,
                                           config._final,
                                           config.heuristicID,
                                           config.phase_I_type);
  // nvtxRangePop();
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  bool good = false;

  if (result.status == 0 || result.status == 2) {
    good = true;
  }

  std::string hID;

  if (config.heuristicID == 0)
    hID = "zero";
  else if (config.heuristicID == 1)
    hID = "no load-balancing";
  else if (config.heuristicID == 2)
    hID = "budget errorest";
  else if (config.heuristicID == 4)
    hID = "target errorest"; // default
  else if (config.heuristicID == 7)
    hID = "estimate budget";
  // else if(config.heuristicID == 8)
  //   hID = "extreme";
  else if (config.heuristicID == 9)
    hID = "aggressive";

  outfile.precision(17);
  outfile << std::fixed << std::scientific << id << "," << hID << ","
          << true_value << "," << epsrel << "," << epsabs << ","
          << result.estimate << "," << result.errorest << "," << result.nregions
          << "," << result.nFinishedRegions << "," << result.status << ","
          << config._final << "," << result.lastPhase << "," << dt.count()
          << std::endl;
  return good;
}

template <typename F, int ndim>
bool
cu_time_and_call_100(std::string id,
                 F integrand,
                 double epsrel,
                 double true_value,
                 char const* algname,
                 std::ostream& outfile,
                 Config config = Config(),
                 quad::Volume<double, ndim>* vol = nullptr)
{
    bool good = false;
    
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-20;

  quad::Pagani<double, ndim> alg;
  
                                         
  for(int i=0; i < 50; i++){      
    auto const t0 = std::chrono::high_resolution_clock::now();
  // nvtxRangePushA("init_host_data");
    cuhreResult const result = alg.integrate(integrand,
                                           epsrel,
                                           epsabs,
                                           vol,
                                           config.outfileVerbosity,
                                           config._final,
                                           config.heuristicID,
                                           config.phase_I_type);
                                    
                                           
  // nvtxRangePop();
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  
  if (result.status == 0 || result.status == 2) {
    good = true;
  }

  std::string hID;

  if (config.heuristicID == 0)
    hID = "zero";
  else if (config.heuristicID == 1)
    hID = "no load-balancing";
  else if (config.heuristicID == 2)
    hID = "budget errorest";
  else if (config.heuristicID == 4)
    hID = "target errorest"; // default
  else if (config.heuristicID == 7)
    hID = "estimate budget";
  // else if(config.heuristicID == 8)
  //   hID = "extreme";
  else if (config.heuristicID == 9)
    hID = "aggressive";

  outfile.precision(17);
  outfile << std::fixed << std::scientific << id << "," << hID << ","
          << true_value << "," << epsrel << "," << epsabs << ","
          << result.estimate << "," << result.errorest << "," << result.nregions
          << "," << result.nFinishedRegions << "," << result.status << ","
          << config._final << "," << result.lastPhase << "," << dt.count()
          << std::endl;
  }
  
  return good;
}

template <typename F, int ndim>
bool
common_header_pagani_time_and_call(std::string alg_id,
                 std::string integ_id, 
                 F integrand,
                 double epsrel,
                 double true_value,
                 double difficulty,
                 char const* algname,
                 std::ostream& outfile,
                 Config config = Config(),
                 quad::Volume<double, ndim>* vol = nullptr)
{
    bool good = false;
    
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-20;

  quad::Pagani<double, ndim> alg;
  
                                         

    auto const t0 = std::chrono::high_resolution_clock::now();
  // nvtxRangePushA("init_host_data");
    cuhreResult const result = alg.integrate(integrand,
                                           epsrel,
                                           epsabs,
                                           vol,
                                           config.outfileVerbosity,
                                           config._final,
                                           config.heuristicID,
                                           config.phase_I_type);
                                    
                                           
  // nvtxRangePop();
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  
  if (result.status == 0 || result.status == 2) {
    good = true;
  }

  std::string hID;



  outfile.precision(17);
  outfile << std::fixed << std::scientific << integ_id << ","    
          << std::scientific << alg_id << "," 
          << difficulty << ","
          << epsrel << "," 
          << epsabs << ","
          << std::scientific << true_value << "," 
          << std::scientific << result.estimate << "," 
          << std::scientific << result.errorest << "," 
          << dt.count() << ","
          << result.status 
          << std::endl;
  
  
  return good;
}