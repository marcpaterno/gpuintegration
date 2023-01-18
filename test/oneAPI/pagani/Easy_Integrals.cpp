#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "catch2/catch.hpp"
#include "oneAPI/integrands.hpp"
#include "oneAPI/pagani/quad/GPUquad/Workspace.dp.hpp"
#include "oneAPI/pagani/quad/quad.h"
#include "oneAPI/pagani/quad/util/Volume.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaUtil.h"
#include "oneAPI/integrands.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

class PTest {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    double res = 15.37 + 0 * x + 0 * y;
    return res;
  }
};

class NTest {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    double res = -15.37 + 0 * x + 0 * y;
    return res;
  }
};

class ZTest {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    return 0. * x * y;
  }
};

TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  PTest integrand;
  constexpr bool use_custom = false;

  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  Workspace<ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;

  auto res = pagani.integrate(integrand, epsrel, epsabs, vol);

  double integral = res.estimate;

  // returns are never precisely equal to 0. and 15.37
  printf("totalEstimate: %.15f\n", integral);
  CHECK(integral == Approx(15.37));
}

TEST_CASE("Genz2 Integral")
{
  double epsrel = 1.e-3;
  double epsabs = 1.0e-12;
  double true_value = 1.286889807581113e+13;
  constexpr int ndim = 6;
  F_2_6D integrand;
  constexpr bool use_custom = false;

  Workspace<ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  auto res = pagani.integrate(integrand, epsrel, epsabs, vol);
  double integral = res.estimate;
  CHECK(integral == Approx(true_value).epsilon(epsrel));
}

TEST_CASE("Genz3 Integral")
{
  double epsrel = 1.e-3;
  double epsabs = 1.0e-12;
  double true_value = 2.2751965817917756076e-10;
  constexpr int ndim = 8;
  F_3_8D integrand;
  constexpr bool use_custom = false;
  Workspace<ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  auto res = pagani.integrate(integrand, epsrel, epsabs, vol);
  double integral = res.estimate;
  CHECK(integral == Approx(true_value).epsilon(epsrel));
}

TEST_CASE("Genz4 Integral")
{
  double epsrel = 1.e-3;
  double epsabs = 1.0e-12;
  double true_value = 1.79132603674879e-06;
  constexpr int ndim = 5;
  F_4_5D integrand;
  constexpr bool use_custom = false;

  Workspace<ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  auto res = pagani.integrate(integrand, epsrel, epsabs, vol);
  double integral = res.estimate;
  CHECK(integral == Approx(true_value).epsilon(epsrel));
}

TEST_CASE("Genz5 Integral")
{
  double epsrel = 1.e-3;
  double epsabs = 1.0e-12;
  double true_value = 2.425217625641885e-06;
  constexpr int ndim = 8;
  F_5_8D integrand;
  constexpr bool use_custom = false;
  Workspace<ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  auto res = pagani.integrate(integrand, epsrel, epsabs, vol);
  double integral = res.estimate;
  CHECK(integral == Approx(true_value).epsilon(epsrel));
}

TEST_CASE("Genz6 Integral")
{
  double epsrel = 1.e-3;
  double epsabs = 1.0e-12;
  double true_value = 1.5477367885091207413e8;
  constexpr int ndim = 6;
  F_6_6D integrand;
  constexpr bool use_custom = false;
  Workspace<ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  auto res = pagani.integrate(integrand, epsrel, epsabs, vol);
  double integral = res.estimate;
  CHECK(integral == Approx(true_value).epsilon(epsrel));
}