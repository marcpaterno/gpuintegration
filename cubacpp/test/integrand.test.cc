#include "catch2/catch.hpp"
#include "cubacpp/cuba_wrapped_integrand.hh"
#include "cubacpp/gsl.hh"
#include <iostream>

TEST_CASE("Scalar 1D integrand can be called", "[integrand]")
{
  auto myfunc = [](double x) { return 2. * x; };
  using functype = decltype(myfunc);
  auto wrapped_integrand = cubacpp::detail::cuba_wrapped_integrand<functype>;
  cubacpp::integration_volume_for_t<functype> myvolume{};
  cubacpp::detail::definite_integral my_integral{&myfunc, &myvolume};
  double res = 0.0;
  int const ndim = 1;
  double const x = 3.5;
  int const ncomp = 1;
  wrapped_integrand(&ndim, &x, &ncomp, &res, &my_integral);
  REQUIRE(res == 7.0);
}

TEST_CASE("Scalar 2D integrand can be called", "[integrand]")
{
  auto myfunc = [](double x, double y) { return x + y; };
  using functype = decltype(myfunc);
  auto wrapped_integrand = cubacpp::detail::cuba_wrapped_integrand<functype>;
  cubacpp::integration_volume_for_t<functype> myvolume{};
  cubacpp::detail::definite_integral my_definite_integral{&myfunc, &myvolume};
  double res = 0.0;
  int const ndim = 2;
  double const x[] = {2.0, -3.0};
  int const ncomp = 1;
  wrapped_integrand(&ndim, x, &ncomp, &res, &my_definite_integral);
  REQUIRE(res == -1.0);
}

TEST_CASE("2D vector 1D integrand can be called", "[integrand]")
{
  auto myfunc = [](double x) { return std::array<double, 2>{{-x, x}}; };
  using functype = decltype(myfunc);
  auto wrapped_integrand = cubacpp::detail::cuba_wrapped_integrand<functype>;
  cubacpp::integration_volume_for_t<functype> myvolume{};
  cubacpp::detail::definite_integral my_definite_integral{&myfunc, &myvolume};
  double res[] = {0.0, 0.0};
  REQUIRE(sizeof(res) == 2 * sizeof(double));
  double const x = 5.0;
  int const ndim = 1;
  int const ncomp = 2;
  wrapped_integrand(&ndim, &x, &ncomp, res, &my_definite_integral);
  REQUIRE(res[0] == -5.0);
  REQUIRE(res[1] == 5.0);
}

TEST_CASE("2D vector 3D integrand can be called", "[integrand]")
{
  auto myfunc = [](double x, double y, double z) {
    return std::array<double, 2>{{x + y, y + z}};
  };
  using functype = decltype(myfunc);
  auto wrapped_integrand = cubacpp::detail::cuba_wrapped_integrand<functype>;
  cubacpp::integration_volume_for_t<functype> myvolume{};
  cubacpp::detail::definite_integral my_definite_integral{&myfunc, &myvolume};
  double res[] = {0.0, 0.0};
  double const x[] = {1.0, 2.0, 3.0};
  int const ndim = sizeof(x) / sizeof(double);
  int const ncomp = sizeof(res) / sizeof(double);
  wrapped_integrand(&ndim, x, &ncomp, res, &my_definite_integral);
  REQUIRE(res[0] == 3.0);
  REQUIRE(res[1] == 5.0);
}

TEST_CASE("2D vector 2D integrand returning std::vector can be called",
          "[integrand]")
{
  auto myfunc = [](double x, double y) {
    return std::vector<double>{{x + y, x - y}};
  };
  using functype = decltype(myfunc);
  auto wrapped_integrand = cubacpp::detail::cuba_wrapped_integrand<functype>;
  cubacpp::integration_volume_for_t<functype> myvolume{};
  cubacpp::detail::definite_integral my_definite_integral{&myfunc, &myvolume};
  double res[] = {0., 0.};
  double const x[] = {3., 2.};
  int const ndim = sizeof(x) / sizeof(double);
  int const ncomp = 2;
  wrapped_integrand(&ndim, x, &ncomp, res, &my_definite_integral);
  REQUIRE(res[0] == 5.0);
  REQUIRE(res[1] == 1.0);
}

TEST_CASE("Test gsl_function wrapper", "[integrand]")
{
  SECTION("minimal example")
  {
    auto f = [](double a) { return a * a + a; };
    auto wrapped_integrand = cubacpp::detail::make_gsl_integrand(&f);
    REQUIRE(wrapped_integrand.function(0.0, wrapped_integrand.params) == 0.0);
    REQUIRE(wrapped_integrand.function(1.0, wrapped_integrand.params) == 2.0);
  }

  SECTION("polynomial")
  {
    for (auto i = 0; i < 10; i++) {
      for (auto j = 0; j < 10; j++) {
        const double x = ((double)i) / 10.0, y = ((double)j) / 10.0,
                     a2 = 2 * x + y, a1 = 3 * x * y - 3,
                     a0 = x / (y + 1) + x * x * 4;
        auto f = [a2, a1, a0](double a) { return a2 * a * a + a1 * a + a0; };
        auto wrapped_integrand = cubacpp::detail::make_gsl_integrand(&f);
        CHECK(wrapped_integrand.function(1, wrapped_integrand.params) ==
              Approx(a2 + a1 + a0));
      }
    }
  }
}
