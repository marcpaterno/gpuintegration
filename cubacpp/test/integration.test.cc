#include "catch2/catch.hpp"
#include "cubacpp/common_results.hh"
#include "cubacpp/cubacpp.hh"
#include "cubacpp/gsl.hh"

#include <array>
#include <cmath>
#include <iostream>

double constexpr pi = 0x1.921fb54442d18p+1;
double scalar_free_function_6integral =
  4.14199; // From Mathematica 11.2 NIntegrate to 6 figures.
double constexpr epsrel = 1.e-3;
double constexpr epsabs = 1.e-12;

// StatefulScalarFunction is an example of a user-defined function to be
// integrated. Such a function is to be written as either a class or a struct.
// It must have const member function operator(), the function call operator,
// which takes one or more doubles (or types that can be converted to doubles).
class StatefulScalarFunction {
public:
  explicit StatefulScalarFunction(double mul) : multiplier(mul){};

  double
  operator()(double x, double y) const
  {
    return multiplier * x * y * (x + y);
  }

private:
  double multiplier;
};

// Scalar-valued free function of one argument.
inline double
scalar_free_function_1(double x)
{
  auto sinx = std::sin(pi * x);
  return sinx * sinx;
}
double constexpr scalar_free_function_1res = 0.5;

// Scalar-valued free function of two arguments.
inline double
scalar_free_function_2(double x, double y)
{
  return 3. * x * y * (x + y);
};
double constexpr scalar_free_function_2res = 1.0;

// Array-valued function of two arguments.
inline std::array<double, 4>
array_free_function(double x, double y)
{
  double const f1 = x + y;
  double const f2 = x * f1;
  double const f3 = y * f1;
  return {{f1, f2, f3, f1}};
}
std::array<double, 4> constexpr array_free_functionres{
  {1, 7. / 12., 7. / 12., 1.}};

class StatefulVectorFunction {
public:
  std::vector<double>
  operator()(double x, double y) const
  {
    auto val = array_free_function(x, y);
    return {val.cbegin(), val.cend()};
  }
};

// scalar_free_function_6 is an example vector-valued function of 6 arguments.
inline double
scalar_free_function_6(double u,
                       double v,
                       double w,
                       double x,
                       double y,
                       double z)
{
  return 44 * (std::sin(pi * u * v) / u) * x * y * std::pow(w, y) *
         std::sin(pi * z * z);
}
double scalar_free_function_6res =
  4.14199; // From Mathematica 11.2 NIntegrate to 6 figures.

// fracerr returns the absolute value of the fractional error.
double
fracerr(double actual, double expected)
{
  return std::abs((actual - expected) / expected);
}

TEST_CASE("cuhre works for vector functions", "[integration][cuhre]")
{
  cubacores(0, 0);
  cubacpp::Cuhre alg;

  SECTION("myvfunc")
  {
    StatefulVectorFunction ff;
    cubacpp::integration_results_v res = alg.integrate(ff, epsrel, epsabs);
    CHECK(res.value[0] == Approx(1.0).epsilon(epsrel));
    CHECK(res.value[1] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[2] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[3] == Approx(1.0).epsilon(epsrel));
    for (std::size_t i = 0; i != 4; ++i) {
      CHECK(fracerr(res.value[i], array_free_functionres[i]) < epsrel);
    }
    CHECK(res.status == 0);
  }
}

TEST_CASE("cuhre works", "[integration][cuhre]")
{
  cubacores(0, 0);
  cubacpp::Cuhre alg;
  SECTION("scalar_free_function_2")
  {
    auto res = alg.integrate(scalar_free_function_2, epsrel, epsabs);
    CHECK(res.value == Approx(scalar_free_function_2res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_2res) < epsrel);
    CHECK(res.neval < 10000);
    CHECK(res.status == 0);
  }
  SECTION("array_free_function")
  {
    auto res = alg.integrate(array_free_function, epsrel, epsabs);
    CHECK(res.value[0] == Approx(1.0).epsilon(epsrel));
    CHECK(res.value[1] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[2] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[3] == Approx(1.0).epsilon(epsrel));
    for (std::size_t i = 0; i != 4; ++i) {
      CHECK(fracerr(res.value[i], array_free_functionres[i]) < epsrel);
    }
    CHECK(res.status == 0);
  }
  SECTION("myfunc")
  {
    StatefulScalarFunction ff{3.0};
    auto res = alg.integrate(ff, epsrel, epsabs);
    CHECK(res.value == Approx(1.0).epsilon(epsrel));
    CHECK(fracerr(res.value, 1.) < epsrel);
    CHECK(res.status == 0);
  }
  SECTION("scalar_free_function_6")
  {
    auto res = alg.integrate(scalar_free_function_6, epsrel, epsabs);
    CHECK(res.value == Approx(scalar_free_function_6res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_6res) < epsrel);
    CHECK(res.status == 0);
  }
}

TEST_CASE("vegas works for vector functions", "[integration][vegas]")
{
  cubacores(0, 0);
  cubacpp::Vegas alg;

  SECTION("myvfunc")
  {
    StatefulVectorFunction ff;
    alg.maxeval = 200 * 1000; // value tweaked to reach convergence
    cubacpp::integration_results_v res = alg.integrate(ff, epsrel, epsabs);
    CHECK(res.value[0] == Approx(1.0).epsilon(epsrel));
    CHECK(res.value[1] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[2] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[3] == Approx(1.0).epsilon(epsrel));
    for (std::size_t i = 0; i != 4; ++i) {
      CHECK(fracerr(res.value[i], array_free_functionres[i]) < epsrel);
    }
    CHECK(res.status == 0);
  }
}

TEST_CASE("vegas works", "[integration][vegas]")
{
  cubacores(0, 0);
  cubacpp::Vegas alg;
  SECTION("scalar_free_function_1")
  {
    auto res = alg.integrate(scalar_free_function_1, epsrel, epsabs);
    CHECK(res.value == Approx(scalar_free_function_1res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_1res) < epsrel);
    CHECK(res.status == 0);
  }
  SECTION("scalar_free_function_2")
  {
    alg.maxeval = 50 * 1000;
    auto res = alg.integrate(scalar_free_function_2, epsrel, epsabs);
    CHECK(res.value == Approx(scalar_free_function_2res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_2res) < epsrel);
    CHECK(res.neval < 15000);
    CHECK(res.status == 0);
  }
  SECTION("array_free_function")
  {
    alg.maxeval = 200 * 1000;
    auto res = alg.integrate(array_free_function, epsrel, epsabs);
    CHECK(res.value[0] == Approx(1.0).epsilon(epsrel));
    CHECK(res.value[1] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[2] == Approx(7. / 12.).epsilon(epsrel));
    for (std::size_t i = 0; i != 3; ++i) {
      CHECK(fracerr(res.value[i], array_free_functionres[i]) < epsrel);
    }
    CHECK(res.status == 0);
  }
  SECTION("myfunc")
  {
    alg.maxeval = 50 * 1000;
    StatefulScalarFunction ff{3.0};
    auto res = alg.integrate(ff, epsrel, epsabs);
    CHECK(res.value == Approx(1.0).epsilon(epsrel));
    CHECK(fracerr(res.value, 1) < epsrel);
    CHECK(res.status == 0);
  }
  SECTION("scalar_free_function_6")
  {
    alg.maxeval = 100 * 1000;
    auto res = alg.integrate(scalar_free_function_6, epsrel, epsabs);
    CHECK(res.value == Approx(scalar_free_function_6res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_6res) < epsrel);
    CHECK(res.status == 0);
  }
}
//
TEST_CASE("suave works for vector functions", "[integration][suave]")
{
  cubacores(0, 0);
  cubacpp::Suave alg;
  //
  SECTION("myvfunc")
  {
    StatefulVectorFunction ff;
    alg.maxeval = 200 * 1000; // value tweaked to reach convergence
    cubacpp::integration_results_v res = alg.integrate(ff, epsrel, epsabs);
    CHECK(res.value[0] == Approx(1.0).epsilon(epsrel));
    CHECK(res.value[1] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[2] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[3] == Approx(1.0).epsilon(epsrel));
    for (std::size_t i = 0; i != 4; ++i) {
      CHECK(fracerr(res.value[i], array_free_functionres[i]) < epsrel);
    }
    CHECK(res.status == 0);
  }
}
//
TEST_CASE("suave works", "[integration][suave]")
{
  cubacores(0, 0);
  cubacpp::Suave alg;
  SECTION("scalar_free_function_1")
  {
    auto res = alg.integrate(scalar_free_function_1, epsrel, epsabs);
    CHECK(res.value == Approx(scalar_free_function_1res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_1res) < epsrel);
    CHECK(res.status == 0);
  }
  SECTION("scalar_free_function_2")
  {
    alg.maxeval = 50 * 1000;
    auto res = alg.integrate(scalar_free_function_2, epsrel, epsabs);
    CHECK(res.value == Approx(scalar_free_function_2res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_2res) < epsrel);
    CHECK(res.neval < 15000);
    CHECK(res.status == 0);
  }
  SECTION("array_free_function")
  {
    alg.maxeval = 50 * 1000;
    auto res = alg.integrate(array_free_function, epsrel, epsabs);
    CHECK(res.value[0] == Approx(1.0).epsilon(epsrel));
    CHECK(res.value[1] == Approx(7. / 12.).epsilon(epsrel));
    CHECK(res.value[2] == Approx(7. / 12.).epsilon(epsrel));
    for (std::size_t i = 0; i != 3; ++i) {
      CHECK(fracerr(res.value[i], array_free_functionres[i]) < epsrel);
    }
    CHECK(res.status == 0);
  }
  SECTION("myfunc")
  {
    alg.maxeval = 50 * 1000;
    StatefulScalarFunction ff{3.0};
    auto res = alg.integrate(ff, epsrel, epsabs);
    CHECK(res.value == Approx(1.0).epsilon(epsrel));
    CHECK(fracerr(res.value, 1.) < epsrel);
    CHECK(res.status == 0);
  }
  SECTION("scalar_free_function_6")
  {
    alg.maxeval = 100 * 1000;
    auto res = alg.integrate(scalar_free_function_6, epsrel, epsabs);
    CHECK(res.value == Approx(scalar_free_function_6res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_6res) < epsrel);
    CHECK(res.status == 0);
  }
}

TEST_CASE("qng works", "[integration][qng]")
{
  cubacpp::QNG qng;
  SECTION("scalar_free_function_1")
  {
    auto res = qng.integrate(scalar_free_function_1, epsrel, epsabs);
    CHECK(res.status == 0);
    CHECK(res.value == Approx(scalar_free_function_1res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_1res) < epsrel);
  }
  SECTION("linear")
  {
    auto res = qng.integrate([](double a) { return a; }, epsrel, epsabs);
    CHECK(res.status == 0);
    CHECK(res.value == Approx(0.5).epsilon(epsrel));
    CHECK(fracerr(res.value, 0.5) < epsrel);
  }
  SECTION("polynomial")
  {
    for (auto i = 0; i < 10; i++) {
      for (auto j = 0; j < 10; j++) {
        const double x = ((double)i) / 10, y = ((double)j) / 10, a2 = 2 * x + y,
                     a1 = 3 * x * y - 3, a0 = x / (y + 1) + x * x * 4;

        auto res = qng.integrate(
          [a2, a1, a0](double a) { return a2 * a * a + a1 * a + a0; },
          epsrel,
          epsabs);
        CHECK(res.status == 0);
        CHECK(res.value == Approx((1.0 / 3.0) * a2 + 0.5 * a1 + a0));
      }
    }
  }
  SECTION("gaussian")
  {
    auto gaussian = [=](double x, double mu, double sigma) {
      const double z = (x - mu) / sigma;
      return 1 / (std::sqrt(2.0 * pi) * sigma) * std::exp(-z * z / 2);
    };

    auto gaussian_integral =
      [=](double xmin, double xmax, double mu, double sigma) {
        const double denom = std::sqrt(2.0) * sigma;
        return 0.5 *
               (std::erf((xmax - mu) / denom) - std::erf((xmin - mu) / denom));
      };

    for (auto i = 0; i < 10; i++) {
      for (auto j = 0; j < 10; j++) {
        const double mu = (((double)i) + 1) / 10.0,
                     sigma = (((double)j) + 1) / 10.0;
        auto res = qng.integrate(
          [&](double x) { return gaussian(x, mu, sigma); }, epsrel, epsabs);
        CHECK(res.status == 0);
        CHECK(res.value == Approx(gaussian_integral(0.0, 1.0, mu, sigma))
                             .epsilon(epsrel)
                             .margin(epsabs));
      }
    }
  }
}

TEST_CASE("qag works", "[integration][qag]")
{
  cubacpp::QAG qag(0.0, 1.0, GSL_INTEG_GAUSS61, 20);
  SECTION("scalar_free_function_1")
  {
    auto res = qag.integrate(scalar_free_function_1, epsrel, epsabs);
    CHECK(res.status == 0);
    CHECK(res.value == Approx(scalar_free_function_1res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_1res) < epsrel);
  }
  SECTION("linear")
  {
    auto res = qag.integrate([](double a) { return a; }, epsrel, epsabs);
    CHECK(res.status == 0);
    CHECK(res.value == Approx(0.5).epsilon(epsrel));
    CHECK(fracerr(res.value, 0.5) < epsrel);
  }
  SECTION("polynomial")
  {
    for (auto i = 0; i < 10; i++) {
      for (auto j = 0; j < 10; j++) {
        const double x = ((double)i) / 10, y = ((double)j) / 10, a2 = 2 * x + y,
                     a1 = 3 * x * y - 3, a0 = x / (y + 1) + x * x * 4;

        auto res = qag.integrate(
          [a2, a1, a0](double a) { return a2 * a * a + a1 * a + a0; },
          epsrel,
          epsabs);
        CHECK(res.status == 0);
        CHECK(res.value == Approx((1.0 / 3.0) * a2 + 0.5 * a1 + a0));
      }
    }
  }
  SECTION("gaussian")
  {
    auto gaussian = [=](double x, double mu, double sigma) {
      const double z = (x - mu) / sigma;
      return 1 / (std::sqrt(2.0 * pi) * sigma) * std::exp(-z * z / 2);
    };

    auto gaussian_integral =
      [=](double xmin, double xmax, double mu, double sigma) {
        const double denom = std::sqrt(2.0) * sigma;
        return 0.5 *
               (std::erf((xmax - mu) / denom) - std::erf((xmin - mu) / denom));
      };

    for (auto i = 0; i < 10; i++) {
      for (auto j = 0; j < 10; j++) {
        const double mu = (((double)i) + 1) / 10.0,
                     sigma = (((double)j) + 1) / 10.0;
        auto res = qag.integrate(
          [&](double x) { return gaussian(x, mu, sigma); }, epsrel, epsabs);
        CHECK(res.status == 0);
        CHECK(res.value == Approx(gaussian_integral(0.0, 1.0, mu, sigma))
                             .epsilon(epsrel)
                             .margin(epsabs));
      }
    }
  }
}

TEST_CASE("cquad works", "[integration][cquad]")
{
  cubacpp::CQUAD cquad;
  SECTION("scalar_free_function_1")
  {
    auto res = cquad.integrate(scalar_free_function_1, epsrel, epsabs);
    CHECK(res.status == 0);
    CHECK(res.value == Approx(scalar_free_function_1res).epsilon(epsrel));
    CHECK(fracerr(res.value, scalar_free_function_1res) < epsrel);
  }
  SECTION("linear")
  {
    auto res = cquad.integrate([](double a) { return a; }, epsrel, epsabs);
    CHECK(res.status == 0);
    CHECK(res.value == Approx(0.5).epsilon(epsrel));
    CHECK(fracerr(res.value, 0.5) < epsrel);
  }
  SECTION("polynomial")
  {
    for (auto i = 0; i < 10; i++) {
      for (auto j = 0; j < 10; j++) {
        const double x = ((double)i) / 10, y = ((double)j) / 10, a2 = 2 * x + y,
                     a1 = 3 * x * y - 3, a0 = x / (y + 1) + x * x * 4;

        auto res = cquad.integrate(
          [a2, a1, a0](double a) { return a2 * a * a + a1 * a + a0; },
          epsrel,
          epsabs);
        CHECK(res.status == 0);
        CHECK(res.value == Approx((1.0 / 3.0) * a2 + 0.5 * a1 + a0));
      }
    }
  }
  SECTION("gaussian")
  {
    auto gaussian = [=](double x, double mu, double sigma) {
      const double z = (x - mu) / sigma;
      return 1 / (std::sqrt(2.0 * pi) * sigma) * std::exp(-z * z / 2);
    };

    auto gaussian_integral =
      [=](double xmin, double xmax, double mu, double sigma) {
        const double denom = std::sqrt(2.0) * sigma;
        return 0.5 *
               (std::erf((xmax - mu) / denom) - std::erf((xmin - mu) / denom));
      };

    for (auto i = 0; i < 10; i++) {
      for (auto j = 0; j < 10; j++) {
        const double mu = (((double)i) + 1) / 10.0,
                     sigma = (((double)j) + 1) / 10.0;
        auto res = cquad.integrate(
          [&](double x) { return gaussian(x, mu, sigma); }, epsrel, epsabs);
        CHECK(res.status == 0);
        CHECK(res.value == Approx(gaussian_integral(0.0, 1.0, mu, sigma))
                             .epsilon(epsrel)
                             .margin(epsabs));
      }
    }
  }
}

TEST_CASE("qawf works", "[integration][qawf]")
{
  // Answer from Mathematica 12.0.0.0
  // Integrate[x * Exp[-x] * Sin[3 x], {x, 3, Infinity}]
  //  = (48 Cos[9] + 11 Sin[9])/(50 E^3)
  //
  // Integrate[x * Exp--x] * Sin[3 x],{x, 0, Infinity}]
  //  = 3/50

  // Note: we intentionally give the *wrong* range to the constructor,
  // to make sure that with_range() is working.
  cubacpp::QAWF qawf(100, 1.0, 3.0, GSL_INTEG_SINE, 5);
  auto igrand = [](double x) { return x * std::exp(-x); };
  SECTION("from3")
  {
    double expected =
      (48 * std::cos(9.0) + 11 * std::sin(9.0)) / (50 * std::exp(3.0));
    auto res = qawf.with_range(3).integrate(igrand, epsabs);
    CHECK(res.status == 0);
    CHECK(res.value == Approx(expected).margin(epsabs));
  }
  SECTION("from0")
  {
    double expected = 3.0 / 50.0;
    auto res = qawf.with_range(0.0).integrate(igrand, epsabs);
    CHECK(res.status == 0);
    CHECK(res.value == Approx(expected).margin(epsabs));
  }
}