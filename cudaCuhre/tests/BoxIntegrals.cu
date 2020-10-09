#include "catch2/catch.hpp"
#include "demos/function.cuh"
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

TEST_CASE("BoxIntegral8_15")
{
  double epsrel = 1.0e-7; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  quad::Cuhre<double, ndim> cuhre(0, nullptr, 0, 0, 1);
  BoxIntegral8_15 integrand;

  std::stringstream outfile;
  std::string id = "BoxIntegral8_15";
  std::string timefile = id + "_time.out";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 8879.851175413485;

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto const result = cuhre.integrate<BoxIntegral8_15>(
    integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);

  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

  //-------------------------------------------------
  // read old timing to compare
  std::ifstream myfile(timefile.c_str());
  double last_recorded_time = 0;

  // output data for analysis
  FinalDataPrint(outfile,
                 id,
                 true_value,
                 epsrel,
                 epsabs,
                 result.estimate,
                 result.errorest,
                 result.nregions,
                 result.status,
                 _final,
                 dt.count(),
                 id + ".csv");

  if (myfile.is_open()) {
    myfile >> last_recorded_time;
    CHECK(dt.count() <= last_recorded_time + 100);
  }

  double const ratio = result.errorest / (epsrel * result.estimate);
  CHECK(ratio <= 1.0);
  double true_err = abs(true_value - result.estimate);
  CHECK(true_err <= result.errorest);

  //-----------------------------------------------------------------------------
  // record new timing if all assertions are passed
  std::stringstream outfileTime;
  outfileTime << dt.count() << std::endl;
  PrintToFile(outfileTime.str(), timefile);
};

TEST_CASE("BoxIntegral8_25")
{
  double epsrel = 1.0e-7; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  quad::Cuhre<double, ndim> cuhre(0, nullptr, 0, 0, 1);
  BoxIntegral8_25 integrand;

  std::stringstream outfile;
  std::string id = "BoxIntegral8_25";
  std::string timefile = id + "_time.out";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 14996089.096112404019;

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto const result = cuhre.integrate<BoxIntegral8_25>(
    integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);

  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

  //-------------------------------------------------
  // read old timing to compare



  // output data for analysis
  /*FinalDataPrint(outfile,
                 id,
                 true_value,
                 epsrel,
                 epsabs,
                 result.estimate,
                 result.errorest,
                 result.nregions,
                 result.status,
                 _final,
                 dt.count(),
                 id + ".csv");*/
  printf("%.20f +- %.20f nregions:%i\n", result.estimate, result.errorest, result.nregions);


  double const ratio = result.errorest / (epsrel * result.estimate);
  CHECK(ratio <= 1.0);
  double true_err = abs(true_value - result.estimate);
  CHECK(true_err <= result.errorest);

};

TEST_CASE("BoxIntegral8_22")
{
  double epsrel = 1.0e-7; // starting error tolerance.
  constexpr double epsabs = 1e-12;
  double lows[] = {0., 0., 0., 0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1., 1., 1., 1.};
  constexpr int ndim = 8;
  quad::Volume<double, ndim> vol(lows, highs);
  quad::Cuhre<double, ndim> cuhre(0, nullptr, 0, 0, 1);
  BoxIntegral8_22 integrand;

  std::stringstream outfile;
  std::string id = "BoxIntegral8_22";
  std::string timefile = id + "_time.out";
  int _final = 1;
  int outfileVerbosity = 0;
  int phase_I_type = 0; // alternative phase 1
  double true_value = 1495369.283757217694;

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto const result = cuhre.integrate<BoxIntegral8_22>(
    integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);

  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

  //-------------------------------------------------
  // read old timing to compare
  //std::ifstream myfile(timefile.c_str());
  //double last_recorded_time = 0;

  // output data for analysis
  FinalDataPrint(outfile,
                 id,
                 true_value,
                 epsrel,
                 epsabs,
                 result.estimate,
                 result.errorest,
                 result.nregions,
                 result.status,
                 _final,
                 dt.count(),
                 id + ".csv");

 /* if (myfile.is_open()) {
    myfile >> last_recorded_time;
    CHECK(dt.count() <= last_recorded_time + 100);
  }*/

  double const ratio = result.errorest / (epsrel * result.estimate);
  CHECK(ratio <= 1.0);
  double true_err = abs(true_value - result.estimate);
  CHECK(true_err <= result.errorest);

  //-----------------------------------------------------------------------------
  // record new timing if all assertions are passed
//std::stringstream outfileTime;
  //outfileTime << dt.count() << std::endl;
  //PrintToFile(outfileTime.str(), timefile);
};