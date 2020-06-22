#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "function.cuh"
#include "quad/quad.h"
#include "quad/util/cudaUtil.h"
#include <iomanip>

#include "quad/GPUquad/Cuhre.cuh"
#include "quad/util/Volume.cuh"
//#include "demo_utils.h"

#include <chrono>
#include <cmath>
#include <iostream>

// From Mathematica 12.1 Integration, symbolic integration over unit hypercube.
// This is the multiplier that gives genz_1_8d an integrated value of 1 over the
// unit hypercube.
using std::sin;
using std::cos;
using std::abs;

double constexpr integral = 6.371054e-01; // Value is approximate
double constexpr normalization = 1./integral;

struct Genz_1abs_5d {
  __device__ __host__
    Genz_1abs_5d () { };

  __device__ __host__ double
    operator() (double v, double w, double x, double y, double z)
  {
    return normalization * abs(cos(4.*v + 5.*w + 6.*x + 7.*y + 8.*z));
  }
};

// Integrate the provided integrand using quad::Cuhre, to the given relative
// error tolerance, writing out timing information to std::cout.
// Return true if the algorithm converged, and false otherwise.

template <typename F>
bool
time_and_call(F integrand, double epsrel, double correct_answer, char const* algname)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-40;

  double lows[] =  {0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1.};
  constexpr int ndim = 5;
  quad::Volume<double, ndim> vol(lows, highs);

  // Why does the integration algorithm need ndim as a template parameter?
  quad::Cuhre<double, ndim> alg(0, nullptr, 0, 0, 1);
 
  auto t0 = std::chrono::high_resolution_clock::now();
  auto res = alg.integrate(integrand, epsrel, epsabs, &vol);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(res.value - correct_answer);
  bool const good = (res.status == 0);
  std::cout << std::scientific
    << algname << '\t'
    << epsrel << '\t';
  if (good) {
    std::cout << res.value << '\t'
      << res.error << '\t'
      << absolute_error << '\t';
  } else {
    std::cout << "NA\tNA\tNA\t";
  }
  std::cout << res.neval << '\t'
    << res.nregions << '\t'
    << dt.count()
    << std::endl;
  return good;
}

int main()
{
  Genz_1abs_5d integrand;
  double epsrel = 1.0e-3;

  std::cout<< "alg\tepsrel\tvalue\terrorest\terror\tneval\tnregions\ttime\n";

  while (time_and_call(integrand, epsrel, 1.0, "gpucuhre")) {
      epsrel /= 5.0;
  }
}

