#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "catch2/catch.hpp"
#include "oneAPI/integrands.hpp"
#include "oneAPI/pagani/quad/GPUquad/Workspace.dp.hpp"
#include "oneAPI/pagani/quad/quad.h"
#include "oneAPI/pagani/quad/util/Volume.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaUtil.h"
#include "oneAPI/integrands.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

class NaN_Integral {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y)
  {
    return NAN + x + y;
  }
};

TEST_CASE("Return NAN")
{
  double epsrel = 1.0e-3; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  constexpr int ndim = 2;
  quad::Volume<double, ndim> vol;
  Workspace<ndim> pagani;
  NaN_Integral integrand;

  auto const result =
    pagani.integrate<NaN_Integral>(integrand, epsrel, epsabs, vol);
  CHECK(std::isnan(result.estimate) == true);
};
