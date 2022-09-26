#include "kokkos/kokkosPagani/quad/Cuhre.cuh"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

template <typename F, int ndim>
bool
time_and_call(std::string id,
              F integrand,
              double epsrel,
              double true_value,
              std::ostream& outfile,
              int heuristicID = 0)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-20;

  Cuhre<double, ndim> pagani;
  auto const t0 = std::chrono::high_resolution_clock::now();
  cuhreResult const result =
    pagani.Integrate(integrand, epsrel, epsabs, heuristicID);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  bool good = false;

  if (result.status == 0 || result.status == 2) {
    good = true;
  }

  outfile.precision(17);
  outfile << std::fixed << std::scientific << id << "," << true_value << ","
          << epsrel << "," << epsabs << "," << result.estimate << ","
          << result.errorest << "," << result.nregions << ","
          << result.nFinishedRegions << "," << result.status << ","
          << dt.count() << std::endl;
  return good;
}