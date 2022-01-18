#include "cuda/cudaPagani/demos/demo_utils.cuh"
#include "cuda/cudaPagani/demos/function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail {
  class GENZ_1_8D {
  public:
    __device__ __host__ float
    operator()(float s,
               float t,
               float u,
               float v,
               float w,
               float x,
               float y,
               float z)
    {
      return cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y +
                 8. * z);
    }
  };
}

int
main()
{
  float epsrel = 1.0e-3;
  float const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 8;
  detail::GENZ_1_8D integrand;

  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 1;
  // configuration.phase_2 = true;
  float true_value = (1. / 315.) * sin(1.) * sin(3. / 2.) * sin(2.) *
                     sin(5. / 2.) * sin(3.) * sin(7. / 2.) * sin(4.) *
                     (sin(37. / 2.) - sin(35. / 2.));

  PrintHeader();
  while (floatIntegrands::cu_time_and_call<detail::GENZ_1_8D, ndim>(
           "8D f1",
           integrand,
           epsrel,
           true_value,
           "gpucuhre",
           std::cout,
           configuration) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  return 0;
}
