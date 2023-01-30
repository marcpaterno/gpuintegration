#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "oneAPI/pagani/quad/GPUquad/Workspace.dp.hpp"
#include <array>

#include "common/integration_result.hh"

class GENZ_2_2D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    double a = 50.;
    double b = .5;

    double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
    double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));

    double val = term_1 * term_2;
    return val;
  }
};

class GENZ_2_6D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    double a = 50.;
    double b = .5;

    double term_1 = 1. / ((1. / pow(a, 2)) + pow(x - b, 2));
    double term_2 = 1. / ((1. / pow(a, 2)) + pow(y - b, 2));
    double term_3 = 1. / ((1. / pow(a, 2)) + pow(z - b, 2));
    double term_4 = 1. / ((1. / pow(a, 2)) + pow(k - b, 2));
    double term_5 = 1. / ((1. / pow(a, 2)) + pow(l - b, 2));
    double term_6 = 1. / ((1. / pow(a, 2)) + pow(m - b, 2));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

TEST_CASE("Correct integration results on various volumes")
{
  double constexpr epsabs = 1.0e-40;
  int verbose = 0;
  int _final = 1;

  SECTION("With std::array")
  {
    GENZ_2_2D integrand;

    constexpr int ndim = 2;
    double epsrel = 1.0e-7;
    Workspace<ndim> alg;
    double true_answer = 23434.02645929748905473389;
    std::array<double, ndim> lows = {0., 0.};
    std::array<double, ndim> highs = {1., 1.};
    quad::Volume<double, ndim> vol(lows, highs);

    auto res = alg.integrate(integrand, epsrel, epsabs, vol);
    double error = fabs(true_answer - res.estimate);
    double relative_error = error / true_answer;

    CHECK(res.status == 0);
    CHECK(relative_error <= epsrel);
    CHECK(error <= res.errorest);
  }

  SECTION("Unit Volume")
  {
    GENZ_2_2D integrand;

    constexpr int ndim = 2;
    double epsrel = 1.0e-7;
    Workspace<ndim> alg;
    double true_answer = 23434.02645929748905473389;
    double lows[] = {0., 0.};
    double highs[] = {1., 1.};
    quad::Volume<double, ndim> vol(lows, highs);

    auto res = alg.integrate(integrand, epsrel, epsabs, vol);
    double error = fabs(true_answer - res.estimate);
    double relative_error = error / true_answer;

    CHECK(res.status == 0);
    CHECK(relative_error <= epsrel);
    CHECK(error <= res.errorest);
  }

  SECTION("Non-Unit Equal ranges")
  {
    // this should be 1/4 of the Unit Volume true_answer
    GENZ_2_2D integrand;

    double epsrel = 1.0e-7;
    constexpr int ndim = 2;
    Workspace<ndim> alg;
    double true_answer = 5858.50661482437226368347;
    double lows[] = {0., 0.};
    double highs[] = {.5, .5};
    quad::Volume<double, ndim> vol(lows, highs);

    auto res = alg.integrate(integrand, epsrel, epsabs, vol);
    double error = fabs(true_answer - res.estimate);
    double relative_error = error / true_answer;

    CHECK(res.status == 0);
    CHECK(relative_error <= epsrel);
    CHECK(error <= res.errorest);
  }

  SECTION("Non-Unit Different ranges")
  {
    GENZ_2_2D integrand;

    double epsrel = 1.0e-7;
    constexpr int ndim = 2;
    Workspace<ndim> alg;
    double true_answer = 11564.50055253929167520255;
    double lows[] = {0., 0.};
    double highs[] = {.5, .75};
    quad::Volume<double, ndim> vol(lows, highs);

    auto res = alg.integrate(integrand, epsrel, epsabs, vol);
    double error = fabs(true_answer - res.estimate);
    double relative_error = error / true_answer;

    CHECK(res.status == 0);
    CHECK(relative_error <= epsrel);
    CHECK(error <= res.errorest);
  }

  SECTION("Non-Unit Different ranges Smaller Space")
  {
    GENZ_2_2D integrand;

    double epsrel = 1.0e-7;
    constexpr int ndim = 2;
    Workspace<ndim> alg;
    double true_answer = 27.01361247915259511387;
    double lows[] = {.6, .65};
    double highs[] = {.8, .9};
    quad::Volume<double, ndim> vol(lows, highs);

    auto res = alg.integrate(integrand, epsrel, epsabs, vol);
    double error = fabs(true_answer - res.estimate);
    double relative_error = error / true_answer;

    CHECK(res.status == 0);
    CHECK(relative_error <= epsrel);
    CHECK(error <= res.errorest);
  }

  SECTION("High Dimension Different ranges")
  {
    GENZ_2_6D integrand;

    double epsrel = 1.0e-6;
    constexpr int ndim = 6;
    Workspace<ndim> alg;
    double true_answer = 5986238682.18309402465820312500;
    double lows[] = {0., 0., 0., 0., 0., 0.};
    double highs[] = {.5, .75, .6, .3, .8, .4};
    quad::Volume<double, ndim> vol(lows, highs);

    auto res = alg.integrate(integrand, epsrel, epsabs, vol);
    double error = fabs(true_answer - res.estimate);
    double relative_error = error / true_answer;

    CHECK(res.status == 0);
    CHECK(relative_error <= epsrel);
    CHECK(error <= res.errorest);
  }
}
