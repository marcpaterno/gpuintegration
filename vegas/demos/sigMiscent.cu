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
#include "cudaCuhre/integrands/sig_miscent.cuh"
#include "vegas/vegas_mcubes.cuh"
#include <limits>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;


int
main()
{
  double const radius_ = 0x1p+0;
  double const zo_low_ = 0x1.999999999999ap-3;
  double const zo_high_ = 0x1.6666666666666p-2;

  integral<GPU> d_integrand;
  constexpr int ndim = 7;
  d_integrand.set_grid_point({zo_low_, zo_high_, radius_});
  Vegas_mcubes<integral<GPU>, ndim>(d_integrand);
  return 0;
}
