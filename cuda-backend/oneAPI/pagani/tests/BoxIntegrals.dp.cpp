#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/catch2/catch.hpp"
#include "oneAPI/pagani/demos/function.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Pagani.dp.hpp"
#include "oneAPI/pagani/quad/quad.h"
#include "oneAPI/pagani/quad/util/Volume.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaUtil.h"
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
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  quad::Pagani<double, ndim> pagani;
  BoxIntegral8_15 integrand;

  std::string id = "BoxIntegral8_15";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 8879.851175413485;

  auto const result = pagani.integrate<BoxIntegral8_15>(
    integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};

TEST_CASE("BoxIntegral8_25")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  quad::Pagani<double, ndim> pagani;
  BoxIntegral8_25 integrand;

  std::string id = "BoxIntegral8_25";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 14996089.096112404019;

  auto const result = pagani.integrate<BoxIntegral8_25>(
    integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};

TEST_CASE("BoxIntegral8_22")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  quad::Pagani<double, ndim> pagani;
  BoxIntegral8_22 integrand;

  std::string id = "BoxIntegral8_22";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 1495369.283757217694;

  auto const result = pagani.integrate<BoxIntegral8_22>(
    integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);

  double true__rel_err = abs(true_value - result.estimate) / true_value;
  CHECK(true__rel_err <= epsrel);
};
