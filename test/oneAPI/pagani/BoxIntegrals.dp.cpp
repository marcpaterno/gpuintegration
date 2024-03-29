#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "catch2/catch.hpp"
#include "common/oneAPI/integrands.hpp"
#include "oneAPI/pagani/quad/GPUquad/Workspace.dp.hpp"
#include "oneAPI/pagani/quad/quad.h"
#include "common/oneAPI/Volume.dp.hpp"
#include "common/oneAPI/cudaUtil.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;
/*
TEST_CASE("BoxIntegral8_15")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  Workspace<ndim> pagani;
  BoxIntegral8_15 integrand;

  std::string id = "BoxIntegral8_15";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 8879.851175413485;
  constexpr bool debug_flag = false;
  auto const result = pagani.integrate<BoxIntegral8_15, debug_flag>(
    integrand, epsrel, epsabs, vol);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};
*/
/*
TEST_CASE("BoxIntegral8_25")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  Workspace<ndim> pagani;
  BoxIntegral8_25 integrand;

  std::string id = "BoxIntegral8_25";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 14996089.096112404019;

  constexpr bool debug_flag = false;
  auto const result = pagani.integrate<BoxIntegral8_25, debug_flag>(
    integrand, epsrel, epsabs, vol);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};
*/
TEST_CASE("BoxIntegral8_22")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  Workspace<ndim> pagani;
  BoxIntegral8_22 integrand;

  std::string id = "BoxIntegral8_22";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 1495369.283757217694;

  constexpr bool debug_flag = false;
  auto const result = pagani.integrate<BoxIntegral8_22, debug_flag>(
    integrand, epsrel, epsabs, vol);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};
