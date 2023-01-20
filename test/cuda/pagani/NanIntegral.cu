#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/demos/function.cuh"
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

class NaN_Integral {
public:
  __device__ double
  operator()(double x, double y)
  {
    return NAN;
  }
};

TEST_CASE("Return NAN")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  constexpr int ndim = 2;
  quad::Volume<double, ndim> vol;
  Workspace<double, ndim> pagani;
  NaN_Integral integrand;

  auto const result = pagani.integrate<NaN_Integral>(integrand, epsrel, epsabs, vol);
  CHECK(std::isnan(result.estimate) == true);
};
