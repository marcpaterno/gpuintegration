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

TEST_CASE("BoxIntegral8_15")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;

  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol;
  Workspace<double, ndim> pagani;
  BoxIntegral8_15 integrand;
  double true_value = 8879.851175413485;

  auto const result = pagani.integrate<BoxIntegral8_15>(
    integrand, epsrel, epsabs, vol);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};

TEST_CASE("BoxIntegral8_25")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol;
  Workspace<double, ndim> pagani;
  BoxIntegral8_25 integrand;
  double true_value = 14996089.096112404019;

  auto const result = pagani.integrate<BoxIntegral8_25>(
    integrand, epsrel, epsabs, vol);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};

TEST_CASE("BoxIntegral8_22")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol;
  Workspace<double, ndim> pagani;
  BoxIntegral8_22 integrand;
  double true_value = 1495369.283757217694;

  auto const result = pagani.integrate<BoxIntegral8_22>(
    integrand, epsrel, epsabs, vol);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};
