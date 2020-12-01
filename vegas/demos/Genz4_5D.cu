#include "catch2/catch.hpp"

#include "../cudaCuhre/quad/util/cudaArray.cuh"
#include "modules/sigma_miscent_y1_scalarintegrand.hh"

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
#include "cudaCuhre/integrands/sig_miscent.cuh"
#include "vegas/vegas_mcubes.cuh"
#include <limits>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;


int
main()
{
  GENZ_4_5D integrand;
  constexpr int ndim = 5;
  Vegas_mcubes<GENZ_4_5D, ndim>(integrand);
  return 0;
}
