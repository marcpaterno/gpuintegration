#include "catch2/catch.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "kokkos/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "kokkos/pagani/quad/GPUquad/Workspace.cuh"
#include "common/integration_result.hh"
#include "common/kokkos/integrands.cuh"
#include "common/kokkos/cudaMemoryUtil.h"
#include "common/kokkos/Volume.cuh"

using numint::integration_result;

class PTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    double res = 15.37;
    return res;
  }
};

class NTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    double res = -15.37;
    return res;
  }
};

class ZTest {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    return 0.;
  }
};

TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  PTest integrand;
  constexpr bool use_custom = true;

  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  Workspace<double, ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;

  integration_result res = pagani.integrate(integrand, epsrel, epsabs, vol);

  double integral = res.estimate;

  // returns are never precisely equal to 0. and 15.37
  CHECK(integral == Approx(15.37));
}

TEST_CASE("Genz2 Integral")
{
  double epsrel = 1.e-3;
  double epsabs = 1.0e-12;
  double true_value = 1.286889807581113e+13;
  constexpr int ndim = 6;
  F_2_6D integrand;
  constexpr bool use_custom = true;

  Workspace<double, ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  integration_result res = pagani.integrate(integrand, epsrel, epsabs, vol);
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
  constexpr bool use_custom = true;
  Workspace<double, ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  integration_result res = pagani.integrate(integrand, epsrel, epsabs, vol);
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
  constexpr bool use_custom = true;

  Workspace<double, ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  integration_result res = pagani.integrate(integrand, epsrel, epsabs, vol);
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
  constexpr bool use_custom = true;
  Workspace<double, ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  integration_result res = pagani.integrate(integrand, epsrel, epsabs, vol);
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
  constexpr bool use_custom = true;
  Workspace<double, ndim, use_custom> pagani;
  quad::Volume<double, ndim> vol;
  integration_result res = pagani.integrate(integrand, epsrel, epsabs, vol);
  double integral = res.estimate;
  CHECK(integral == Approx(true_value).epsilon(epsrel));
}