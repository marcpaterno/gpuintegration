#include "catch2/catch.hpp"
#include "cubacpp/integrand_traits.hh"
#include "cubacpp/integration_result.hh"

#include <iostream>
#include <type_traits>
#include <vector>

using cubacpp::integrand_traits;
using cubacpp::integration_result;
using cubacpp::integration_results;
using cubacpp::integration_results_v;
using cubacpp::detail::default_integration_results;
using std::is_same_v;
using std::vector;

template <std::size_t N>
using darray = std::array<double, N>;

double
f1(double)
{
  return 1.0;
}
double
f2(double, double)
{
  return 2.0;
}
double
f3(double, double, double)
{
  return 3.0;
}

TEST_CASE("scalar functions")
{

  SECTION("free function")
  {
    using itraits1 = integrand_traits<decltype(f1)>;
    static_assert(itraits1::ndim == 1UL);
    static_assert(is_same_v<itraits1::function_return_type, double>);
    static_assert(is_same_v<itraits1::arg_type, darray<1>>);
    static_assert(
      is_same_v<decltype(default_integration_results(f1)), integration_result>);

    using itraits2 = integrand_traits<decltype(f2)>;
    static_assert(itraits2::ndim == 2UL);
    static_assert(is_same_v<itraits2::function_return_type, double>);
    static_assert(is_same_v<itraits2::arg_type, darray<2>>);
    static_assert(
      is_same_v<decltype(default_integration_results(f2)), integration_result>);

    using itraits3 = integrand_traits<decltype(f3)>;
    static_assert(itraits3::ndim == 3UL);
    static_assert(is_same_v<itraits3::function_return_type, double>);
    static_assert(is_same_v<itraits3::arg_type, darray<3>>);
    static_assert(
      is_same_v<decltype(default_integration_results(f3)), integration_result>);
  }

  SECTION("callable object")
  {
    struct SF4 {
      double
      operator()(double, double, double, double) const
      {
        return 3;
      }
    };

    using itraits1 = integrand_traits<SF4>;
    static_assert(itraits1::ndim == 4UL);
    static_assert(is_same_v<itraits1::function_return_type, double>);
    static_assert(is_same_v<itraits1::arg_type, darray<4>>);
    SF4 f;
    static_assert(
      is_same_v<decltype(default_integration_results(f)), integration_result>);
  }
}

darray<7>
g2(double, double)
{
  return {};
}
TEST_CASE("array functions")
{
  using arry = darray<7>;
  SECTION("free functions")
  {
    using itraits1 = integrand_traits<decltype(g2)>;
    static_assert(itraits1::ndim == 2UL);
    static_assert(is_same_v<itraits1::function_return_type, arry>);
    static_assert(is_same_v<itraits1::arg_type, darray<2>>);
    static_assert(is_same_v<decltype(default_integration_results(g2)),
                            integration_results<7UL>>);
  }

  SECTION("callable object")
  {
    using arry = std::array<double, 10>;
    struct A10SF2 {
      arry
      operator()(double, double)
      {
        return {};
      }
    };
    using itraits1 = integrand_traits<A10SF2>;
    static_assert(itraits1::ndim == 2UL);
    static_assert(is_same_v<itraits1::function_return_type, arry>);
    static_assert(is_same_v<itraits1::arg_type, darray<2>>);
    A10SF2 f;
    static_assert(is_same_v<decltype(default_integration_results(f)),
                            integration_results<10UL>>);
  };
}

std::vector<double>
ff(double, double)
{
  return std::vector<double>(10, 1.5);
}

TEST_CASE("vector functions")
{
  SECTION("free functions")
  {
    using itraits1 = integrand_traits<decltype(ff)>;
    static_assert(itraits1::ndim == 2UL);
    static_assert(is_same_v<itraits1::function_return_type, vector<double>>);
    static_assert(is_same_v<itraits1::arg_type, darray<2>>);
    auto ires = default_integration_results(ff);
    REQUIRE(ires.value.size() == 10UL);
    REQUIRE(ires.error.size() == 10UL);
    REQUIRE(ires.prob.size() == 10UL);

    static_assert(is_same_v<decltype(default_integration_results(ff)),
                            integration_results_v>);
  }
  SECTION("callable object")
  {
    struct V5SF4 {
      std::vector<double>
      operator()(double, double, double, double)
      {
        return std::vector<double>(5, 2.5);
      }
    };
    using itraits1 = integrand_traits<V5SF4>;
    static_assert(itraits1::ndim == 4UL);
    static_assert(is_same_v<itraits1::function_return_type, vector<double>>);
    static_assert(is_same_v<itraits1::arg_type, darray<4>>);
    V5SF4 f;
    static_assert(is_same_v<decltype(default_integration_results(f)),
                            integration_results_v>);
  }
}
