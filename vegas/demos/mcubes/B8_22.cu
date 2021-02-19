#include "catch2/catch.hpp"
#include "../cudaCuhre/quad/util/cudaArray.cuh"
#include "utils/str_to_doubles.hh"
#include <chrono>
#include <iostream>
#include <vector>

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

// GPU integrator headers
#include "cudaCuhre/demos/function.cuh"
#include "vegas/vegas_mcubes.cuh"
#include <limits>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;


int
main()
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  BoxIntegral8_22 integrand;
  constexpr int ndim = 8;
  auto const t0 = std::chrono::high_resolution_clock::now();
  Vegas_mcubes<BoxIntegral8_22, ndim>(integrand);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  std::cout<<"Time in ms:"<<dt.count()<<"\n";
  return 0;
}
