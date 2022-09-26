#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "oneAPI/quad/Workspace.h"
#include "oneAPI/quad/util/Volume.h"

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

struct Config {
  Config(int verbosity, int heuristic, int phaseT, int deviceNum, int finFlag)
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
  std::cout << "alg, id, ndim, warp_size, epsrel, epsabs, value, estimate, "
               "errorest, regions, finished_regions, "
               "time, status\n";
}

template <typename F, size_t ndim, int warp_size = 32>
bool
time_and_call_pagani(sycl::queue& q,
                     std::string alg_id,
                     std::string integ_id,
                     F& integrand,
                     double true_val,
                     double* lows,
                     double* highs,
                     double epsrel,
                     double epsabs)
{

  bool success = false;
  using Milliseconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  size_t divs_per_dim = 2;

  if (ndim < 5)
    divs_per_dim = 4;
  else if (ndim <= 10)
    divs_per_dim = 2;
  else
    divs_per_dim = 1;
  size_t num_starting_regs = pow((double)divs_per_dim, (double)ndim);

  for (int run = 0; run < 11; ++run) {
    Workspace<ndim> workspace(q);
    auto const t0 = std::chrono::high_resolution_clock::now();
    Sub_regions<ndim> regions(q, divs_per_dim);

    auto res = workspace.template integrate<F, warp_size>(
      q, integrand, lows, highs, regions, epsrel, epsabs);
    Milliseconds dt = std::chrono::high_resolution_clock::now() - t0;
    std::cout.precision(17);
    if (run != 0)
      std::cout << std::fixed << std::scientific << alg_id << "," << integ_id
                << "," << ndim << "," << warp_size << "," << epsrel << ","
                << epsabs << "," << std::scientific << true_val << ","
                << std::scientific << res.estimate << "," << std::scientific
                << res.errorest << "," << res.nregions << ","
                << res.nFinishedRegions << "," << dt.count() << ","
                << res.status << std::endl;

    success = (res.status == 0);
  }

  return success;
}
