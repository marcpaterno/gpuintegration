#include "catch2/catch.hpp"
#include "cubacpp/cubacpp.hh"
#include "cubacpp/cuhre.hh"
#include "cubacpp/integration_volume.hh"
#include "cubacpp/suave.hh"
#include "cubacpp/vegas.hh"

#include <array>

// Notation: the CUBA routines integrate from u=0 to u=1.
// User code expect to integrate from x = a to x=b.
// The IntegrationVolume::transform function turns 'u' into 'x'.

TEST_CASE("volume")
{
  SECTION("1d")
  {
    using volume_t = cubacpp::IntegrationVolume<1>;
    using array_t = std::array<double, 1>;
    using e_array = cubacpp::array<1>;

    volume_t unit{};
    CHECK(unit.jacobian() == 1.0);

    array_t u0{0.0};
    auto x0 = unit.transform(u0);
    CHECK(x0 == u0);

    array_t u1{0.25};
    auto x1 = unit.transform(u1);
    CHECK(x1 == u1);

    array_t u2{1.0};
    auto x2 = unit.transform(u2);
    CHECK(x2 == u2);

    e_array low{-2.0};
    e_array high{2.0};
    volume_t centered{low, high};
    CHECK(centered.jacobian() == 4.0);
    x0 = centered.transform(u0);
    CHECK(x0 == array_t{-2.0});
    x1 = centered.transform(u1);
    CHECK(x1 == array_t{-1.0});
    x2 = centered.transform(u2);
    CHECK(x2 == array_t{2.0});
  }
  SECTION("2d")
  {
    using volume_t = cubacpp::IntegrationVolume<2>;
    using array_t = std::array<double, 2>;
    using e_array = cubacpp::array<2>;

    volume_t unit{};
    CHECK(unit.jacobian() == 1.0);

    array_t u0{0.0, 0.0};
    array_t u1{0.25, 0.5};
    array_t u2{1.0, 1.0};

    auto x0 = unit.transform(u0);
    CHECK(x0 == u0);
    auto x1 = unit.transform(u1);
    CHECK(x1 == u1);
    auto x2 = unit.transform(u2);
    CHECK(x2 == u2);

    e_array low{-2.0, -3.0};
    e_array high{4.0, 7.0};
    volume_t offcenter{low, high};
    CHECK(offcenter.jacobian() == 60.0);
    x0 = offcenter.transform(u0);
    CHECK(x0 == array_t{-2.0, -3.0});
    x1 = offcenter.transform(u1);
    CHECK(x1 == array_t{-0.5, 2.0});
    x2 = offcenter.transform(u2);
    CHECK(x2 == array_t{4.0, 7.0});
  }
}

// fracerr returns the absolute value of the fractional error.
inline double
fracerr(double actual, double expected)
{
  return std::abs((actual - expected) / expected);
}

class Integrand2D {
private:
public:
  Integrand2D(){};

  double
  operator()(double x, double y) const
  {
    return x * x + x * std::sin(7 * y);
  }
};

TEST_CASE("default two-d volume")
{
  // Answer from Mathematica 11.3.0.0
  //   F[x_, y_] := x^2 + x  Sin [7 y]
  //   NIntegrate[F[x, y], {x, 0, 1}, {y, 0, 1}, PrecisionGoal -> 10,
  // WorkingPrecision -> 20]
  double answer = 0.35091174373737309007;
  Integrand2D my_integrand;

  SECTION("cuhre")
  {
    double const epsrel = 1.0e-6;
    double const epsabs = 1.0e-6;
    cubacpp::Cuhre alg;
    auto res = alg.integrate(my_integrand, epsrel, epsabs);
    CHECK(res.value == Approx(answer).epsilon(epsrel));
    CHECK(fracerr(res.value, answer) < epsrel);
    CHECK(res.status == 0);
  }

  SECTION("vegas")
  {
    double const epsrel = 1.0e-3;
    double const epsabs = 1.0e-3;
    cubacpp::Vegas alg;
    alg.maxeval = 100 * 1000;
    auto res = alg.integrate(my_integrand, epsrel, epsabs);
    CHECK(res.value == Approx(answer).epsilon(epsrel));
    CHECK(fracerr(res.value, answer) < epsrel);
    CHECK(res.status == 0);
  }

  SECTION("suave")
  {
    double const epsrel = 1.0e-4;
    double const epsabs = 1.0e-4;
    cubacpp::Suave alg;
    alg.maxeval = 1000 * 1000;
    auto res = alg.integrate(my_integrand, epsrel, epsabs);
    CHECK(res.value == Approx(answer).epsilon(epsrel));
    CHECK(fracerr(res.value, answer) < epsrel);
    CHECK(res.status == 0);
  }
}

TEST_CASE("non-default two-d volume")
{
  // Answer from Mathematica 11.3.0.0
  // F[x_, y_] := x^2 + x  Sin [7 y]
  //  NIntegrate[F[x, y], {x, 1, 5}, {y, -7, 1}, PrecisionGoal -> 10,
  // WorkingPrecision -> 20]
  double answer = 329.88956430563866514;
  Integrand2D my_integrand;
  cubacpp::array<2> lows = {1.0, -7.0};
  cubacpp::array<2> highs = {5.0, 1.0};
  cubacpp::integration_volume_for_t<Integrand2D> vol(lows, highs);
  CHECK(vol.jacobian() == 4.0 * 8.0);

  SECTION("cuhre")
  {
    cubacpp::Cuhre alg;
    double const epsrel = 1.0e-6;
    double const epsabs = 1.0e-6;
    auto res = alg.integrate(my_integrand, epsrel, epsabs, vol);
    CHECK(res.value == Approx(answer).epsilon(epsrel));
    CHECK(fracerr(res.value, answer) < epsrel);
    CHECK(res.status == 0);
  }

  SECTION("vegas")
  {
    cubacpp::Vegas alg;
    alg.maxeval = 100 * 1000;
    double const epsrel = 1.0e-3;
    double const epsabs = 1.0e-3;
    auto res = alg.integrate(my_integrand, epsrel, epsabs, vol);
    CHECK(res.value == Approx(answer).epsilon(epsrel));
    CHECK(fracerr(res.value, answer) < epsrel);
    CHECK(res.status == 0);
  }

  SECTION("suave")
  {
    cubacpp::Suave alg;
    alg.maxeval = 100 * 1000;
    double const epsrel = 1.0e-3;
    double const epsabs = 1.0e-3;
    auto res = alg.integrate(my_integrand, epsrel, epsabs, vol);
    CHECK(res.value == Approx(answer).epsilon(epsrel));
    CHECK(fracerr(res.value, answer) < epsrel);
    CHECK(res.status == 0);
  }
}
