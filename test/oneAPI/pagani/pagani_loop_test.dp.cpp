#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "catch2/catch.hpp"
#include "oneAPI/pagani/quad/GPUquad/Pagani.dp.hpp"
#include <iostream>
#include <math.h>

struct Fcn {
  double const normalization =
    6.0 / (-sin(1.0) - sin(2.0) + sin(4.0) + sin(5.0) - sin(6.0));

  double
  operator()(double x, double y, double z) const
  {
    return normalization * sycl::cos(x + 2. * y + 3 * z);
  }
};

int
main()
{
  quad::Pagani<double, 3> algorithm;
  double const epsrel = 1.0e-3;
  double const epsabs = 1.0e-15;
  Fcn fcn;

  for (int i = 1; i <= 10000; ++i) {
    auto res = algorithm.integrate(fcn, epsrel / i, epsabs);
    if (i % 100 == 0) {
      std::cout << i << ' ' << res << '\n';
    }
  }
}
