#include "../quad/GPUquad/Cuhre.cuh"
#include "catch2/catch.hpp"

#include "genz_1abs_5d.cuh"

TEST_CASE("genz_1abs_5d"){
  SECTION("decreasing epsrel results in decreasing error estimate"){
    double epsrel = 1.0e-3; // starting error tolerance.

double constexpr epsabs = 1.0e-40;

double lows[] = {0., 0., 0., 0., 0.};
double highs[] = {1., 1., 1., 1., 1.};
constexpr int ndim = 5;
quad::Volume<double, ndim> vol(lows, highs);
quad::Cuhre<double, ndim> alg(0, nullptr, 0, 0, 1);

Genz_1abs_5d integrand;

auto const res = alg.integrate(integrand, epsrel, epsabs, &vol);
CHECK(res.status == true);
double const ratio = res.errorest / (epsrel * res.estimate);
CHECK(ratio <= 1.0);
}
}
;
