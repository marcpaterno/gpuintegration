#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>

#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <CL/sycl.hpp>
// #include <dpct/dpct.hpp>

#include "oneAPI/pagani/quad/GPUquad/Workspace.dp.hpp"
#include <string>
// #include <oneapi/mkl.hpp>
// #include "oneapi/mkl/stats.hpp"

#include "common/oneAPI/cuhreResult.dp.hpp"
#include "common/oneAPI/Volume.dp.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

// using namespace quad;

class PTest {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    double res = 15.37;
    return res;
  }
};

class NTest {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    double res = -15.37;
    return res;
  }
};

class ZTest {
public:
  double
  operator()(double x, double y)
  {
    return 0.;
  }
};

#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

using namespace quad;

namespace detail {
  class GENZ_3_3D {
  public:
    double
    operator()(double x, double y, double z)
    {
      return pow(1 + 3 * x + 2 * y + z, -4);
    }
  };
}

TEST_CASE("Positive Value Function")
{
  constexpr int ndim = 2;
  size_t numRegions = 16;
  PTest integrand;
  size_t maxIters = 1;
  int heuristicID = 0;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  Workspace<ndim> pagani;
  quad::Volume<double, ndim> vol;
  constexpr bool debug = false;
  numint::integration_result res =
    pagani.integrate<PTest, debug>(integrand, epsrel, epsabs, vol);

  double integral = res.estimate;
  double error = res.errorest;
  CHECK(Approx(15.37) == integral);
}

TEST_CASE("Negative Value Function")
{
  constexpr int ndim = 2;
  size_t numRegions = 16;
  NTest integrand;
  size_t maxIters = 1;
  int heuristicID = 0;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  Workspace<ndim> pagani;
  quad::Volume<double, ndim> vol;
  constexpr bool debug = false;
  numint::integration_result res =
    pagani.integrate<NTest, debug>(integrand, epsrel, epsabs, vol);

  double integral = res.estimate;
  double error = res.errorest;
  CHECK(Approx(-15.37) == integral);
}

TEST_CASE("Zero Value Function2")
{
  constexpr int ndim = 2;
  size_t numRegions = 16;
  ZTest integrand;
  size_t maxIters = 1;
  int heuristicID = 0;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  quad::Volume<double, ndim> vol;

  Workspace<ndim> pagani;
  constexpr bool debug = false;
  numint::integration_result res =
    pagani.integrate<ZTest, debug>(integrand, epsrel, epsabs, vol);

  double integral = res.estimate;
  double error = res.errorest;
  CHECK(Approx(0.) == 0.);
}