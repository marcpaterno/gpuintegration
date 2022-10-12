#include "kokkos/pagani/quad/Cuhre.cuh"
#include "catch.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class PTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    return 15.37 + 0 * x * y;
  }
};

class NTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    return -15.37 + 0 * x * y;
  }
};

class ZTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    return 0. + 0 * x * y;
  }
};

TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  double hRegsIntegral[16] = {0.};
  double hRegsError[16] = {0.};
  double hRegs[16 * ndim] = {0.};
  double hRegsLength[16 * ndim] = {0.};

  size_t numRegions = 16;
  PTest integrand;
  size_t maxIters = 1;
  int heuristicID = 0;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  Cuhre<double, ndim> cuhre(
    hRegsIntegral, hRegsError, hRegs, hRegsLength, numRegions);
  cuhre.Integrate<PTest>(integrand, epsrel, epsabs, heuristicID, maxIters);

  double firstEstimate = hRegsIntegral[0];
  double totalEstimate = firstEstimate;
  double totalErrorEst = 0.;
  bool nonZeroErrFound = false;
  bool diffIntegralFound = false;

  SECTION("Sub-Regions Have the same Integral Estimate")
  {
    for (size_t regID = 1; regID < numRegions; regID++) {
      diffIntegralFound = hRegsIntegral[regID] == firstEstimate ? false : true;
      nonZeroErrFound = hRegsError[regID] >= 0.00000000000001 ? true : false;
      totalEstimate += hRegsIntegral[regID];
      totalErrorEst += hRegsError[regID];
    }

    CHECK(diffIntegralFound == false);
  }

  CHECK(totalEstimate == Approx(15.37));
  CHECK(nonZeroErrFound == false);
  CHECK(totalErrorEst <= 0.00000000000001);
}

TEST_CASE("Constant Negative Value Function")
{
  constexpr int ndim = 2;
  double hRegsIntegral[16] = {0.};
  double hRegsError[16] = {0.};
  double hRegs[16 * ndim] = {0.};
  double hRegsLength[16 * ndim] = {0.};
  size_t numRegions = 16;
  NTest integrand;
  size_t maxIters = 1;
  int heuristicID = 0;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  Cuhre<double, 2> cuhre(
    hRegsIntegral, hRegsError, hRegs, hRegsLength, numRegions);
  cuhre.Integrate<NTest>(integrand, epsrel, epsabs, heuristicID, maxIters);

  double firstEstimate = hRegsIntegral[0];
  double totalEstimate = firstEstimate;
  double totalErrorEst = 0.;
  bool nonZeroErrFound = false;
  bool diffIntegralFound = false;

  SECTION("Sub-Regions Have the same Integral Estimate")
  {
    for (size_t regID = 1; regID < numRegions; regID++) {
      diffIntegralFound = hRegsIntegral[regID] == firstEstimate ? false : true;
      nonZeroErrFound = hRegsError[regID] >= 0.00000000000001 ? true : false;
      totalEstimate += hRegsIntegral[regID];
      totalErrorEst += hRegsError[regID];
    }

    CHECK(diffIntegralFound == false);
  }

  // returns are never precisely equal to 0. and -15.37
  CHECK(totalEstimate == Approx(-15.37));
  CHECK(nonZeroErrFound == false);
  CHECK(totalErrorEst <= 0.00000000000001);
}

TEST_CASE("Constant Zero Value Function")
{
  constexpr int ndim = 2;
  double hRegsIntegral[16] = {0.};
  double hRegsError[16] = {0.};
  double hRegs[16 * ndim] = {0.};
  double hRegsLength[16 * ndim] = {0.};
  size_t numRegions = 16;
  ZTest integrand;
  size_t maxIters = 1;
  int heuristicID = 0;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  Cuhre<double, 2> cuhre(
    hRegsIntegral, hRegsError, hRegs, hRegsLength, numRegions);
  cuhre.Integrate<ZTest>(integrand, epsrel, epsabs, heuristicID, maxIters);

  double firstEstimate = hRegsIntegral[0];
  double totalEstimate = firstEstimate;
  double totalErrorEst = 0.;
  bool nonZeroErrFound = false;
  bool diffIntegralFound = false;

  SECTION("Sub-Regions Have the same Integral Estimate")
  {
    for (size_t regID = 1; regID < numRegions; regID++) {
      diffIntegralFound = hRegsIntegral[regID] == firstEstimate ? false : true;
      nonZeroErrFound = hRegsError[regID] >= 0.00000000000001 ? true : false;
      totalEstimate += hRegsIntegral[regID];
      totalErrorEst += hRegsError[regID];
    }

    CHECK(diffIntegralFound == false);
  }

  CHECK(Approx(totalEstimate) == 0.0);
  CHECK(nonZeroErrFound == false);
  CHECK(totalErrorEst <= 0.00000000000001);
}
