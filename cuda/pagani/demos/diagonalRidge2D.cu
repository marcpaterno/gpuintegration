#include "cuda/pagani/demos/demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

__host__ __device__ double
inline
diagonal_ridge2D(double u, double v)
{
  double k = 0.01890022674239546529975841;
  return 4 * k * u * u / (.01 + pow(u - v - (1. / 3.), 2));
}

class Diagonal_ridge2D {
public:
  // correct answer: 1 on integration volume (-1,1)

  __device__ __host__ double
  operator()(double u, double v)
  {
    return diagonal_ridge2D(u, v);
  }
};

int
main()
{
  double epsrel = 1e-3;
  double true_value = 1.;

  PrintHeader();

  constexpr int ndim = 2;

  Config configuration;
  configuration.outfileVerbosity = 0;
  // configuration.heuristicID = 4;
  double lows[] = {-1., -1.}; // original bounds
  double highs[] = {1., 1.};
  quad::Volume<double, ndim> vol(lows, highs);

  Diagonal_ridge2D integrand;

  cu_time_and_call("Diagonal_Ridge2D",
                   integrand,
                   epsrel,
                   true_value,
                   "gpucuhre",
                   std::cout,
                   configuration,
                   &vol);
}
