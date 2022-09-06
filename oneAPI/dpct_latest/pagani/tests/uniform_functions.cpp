#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>

#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "oneAPI/dpct_latest/pagani/quad/GPUquad/Workspace.dp.hpp"
#include <string>
//#include <oneapi/mkl.hpp>
//#include "oneapi/mkl/stats.hpp"

#include "oneAPI/dpct_latest/pagani/quad/util/cuhreResult.dp.hpp"
#include "oneAPI/dpct_latest/pagani/quad/util/Volume.dp.hpp"



#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

//using namespace quad;

class PTest {
public:
  double
  operator()(double x, double y)
  { double res = 15.37;
    return res;
  }
};

class NTest {
public:
  double
  operator()(double x, double y)
  {
    double res = -15.37;
    return res;
  }
};

class ZTest {
public:
  double
  operator()(double x, double y)
  {
    return 0.;
  }
};


 

TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  size_t numRegions = 16;
  NTest integrand;
  size_t maxIters = 1;
  int heuristicID = 0;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  std::cout<<"before creating workspace\n";
  Workspace<ndim> pagani;
  std::cout<<"after workspace\n";
  quad::Volume<double, ndim> vol;
  std::cout<<"After volume"<<std::endl;
  cuhreResult res = pagani.integrate<NTest>(integrand, epsrel, epsabs, vol);
    
  double integral = res.estimate;
  double error = res.errorest;

  // returns are never precisely equal to 0. and 15.37
  printf("ttotalEstimate:%.15f\n", integral);
  CHECK(Approx(-15.37) == integral);
}


 /*TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  size_t numRegions = 16;
  ZTest integrand;
  size_t maxIters = 1;
  int heuristicID = 0;
  double epsrel = 1.0e-3;
  double epsabs = 1.0e-12;
  quad::Volume<double, ndim> vol;
    
  Workspace<ndim> pagani;
  cuhreResult res = pagani.integrate<ZTest>(integrand, epsrel, epsabs, vol);

  double integral = res.estimate;
  double error = res.errorest;

  // returns are never precisely equal to 0. and 15.37
  printf("ttotalEstimate:%.15f\n", integral);
  CHECK(Approx(0.) == 0.);
}*/