#include "../quad/GPUquad/Cuhre.cuh"
#include "catch2/catch.hpp"
#include "../quad/quad.h" // for cuhreResult

#include "genz_1abs_5d.cuh"
#include "fun6.cuh"

TEST_CASE("fun6")
{
  SECTION("decreasing epsrel results in non-increasing error estimate")
  {
    // We start with a very large error tolerance, and will
    // repeatedly decrease the tolerance.
    double epsrel = 1.0e-3;

    double constexpr epsabs = 1.0e-40;

    double lows[] =  {0., 0., 0., 0., 0., 0.};
    double highs[] = {1., 1., 1., 1., 1., 1.};
    constexpr int ndim = 6;
    quad::Volume<double, ndim> vol(lows, highs);
    quad::Cuhre<double, ndim> alg(0, nullptr, 0, 0, 1);

    double previous_error_estimate = 1.0; // larger than ever should be returned

    while (epsrel > 1.0e-6) {
      cuhreResult const res = alg.integrate(fun6, epsrel, epsabs, &vol);
      // The integration should have converged.
      CHECK(res.status);

      // The fractional error error estimate should be
      // no larger than the specified fractional error
      // tolerance
      CHECK(res.errorest/res.estimate <= epsrel);

      // The error estimate should be no larger than the previous iteration.
      CHECK(res.errorest <= previous_error_estimate);

      // Prepare for the next loop.
      previous_error_estimate = res.errorest;
      epsrel /= 2.0;
    }
  }
};

TEST_CASE("genz_1abs_5d")
{
  SECTION("decreasing epsrel results in non-increasing error estimate")
  {
    // We start with a very large error tolerance, and will
    // repeatedly decrease the tolerance.
    double epsrel = 1.0e-3;

    double constexpr epsabs = 1.0e-40;

    double lows[] = {0., 0., 0., 0., 0.};
    double highs[] = {1., 1., 1., 1., 1.};
    constexpr int ndim = 5;
    quad::Volume<double, ndim> vol(lows, highs);
    quad::Cuhre<double, ndim> alg(0, nullptr, 0, 0, 1);

    Genz_1abs_5d integrand;
    double previous_error_estimate = 1.0; // larger than ever should be returned

    while (epsrel > 1.0e-6) {
      cuhreResult const res = alg.integrate(integrand, epsrel, epsabs, &vol);
      // The integration should have converged.
      CHECK(res.status);

      // The fractional error error estimate should be
      // no larger than the specified fractional error
      // tolerance
      CHECK(res.errorest/res.estimate <= epsrel);

      // The error estimate should be no larger than the previous iteration.
      CHECK(res.errorest <= previous_error_estimate);

      // Prepare for the next loop.
      previous_error_estimate = res.errorest;
      epsrel /= 2.0;
    }
  }
};

