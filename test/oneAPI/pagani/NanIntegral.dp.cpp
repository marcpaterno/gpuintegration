#define CATCH_CONFIG_MAIN 
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
//#include <dpct/dpl_utils.hpp>
#include "catch2/catch.hpp"

class NaN_Integral {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    return NAN;
  }
};

TEST_CASE("Return NAN")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  constexpr int ndim = 2;

  quad::Pagani<double, ndim> pagani;
  NaN_Integral integrand;

  auto const result = pagani.integrate<NaN_Integral>(integrand, epsrel, epsabs);
  CHECK(std::isnan(result.estimate) == true);
};
