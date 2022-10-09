#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/demos/function.cuh"
#include "cuda/pagani/quad/GPUquad/Sample.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/mem_util.cuh"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
#include "cuda/pagani/quad/util/custom_functions.cuh"
#include "cuda/pagani/quad/util/thrust_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "cuda/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"
#include "cuda/pagani/quad/util/cuhreResult.cuh"
#include "cuda/pagani/quad/util/Volume.cuh"

using namespace quad;

class PTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    double res = 15.37;
    return res;
  }
};

class NTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    double res = -15.37;
    return res;
  }
};

class ZTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    return 0.;
  }
};

TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  PTest integrand;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  constexpr bool use_custom = false;
  Workspace<double, ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;

  cuhreResult res = pagani.integrate(integrand, epsrel, epsabs, vol);

  double integral = res.estimate;
  double error = res.errorest;

  // returns are never precisely equal to 0. and 15.37
  printf("ttotalEstimate:%.15f %.15f\n", integral, error);
  CHECK(integral == Approx(15.37));
}
