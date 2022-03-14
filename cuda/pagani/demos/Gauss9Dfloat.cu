#include "cuda/pagani/demos/demo_utils.cuh"
#include "cuda/pagani/demos/function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;
namespace detail {
  class Gauss9D {
  public:
    __device__ __host__ double
    operator()(double x,
               double y,
               double z,
               double k,
               double l,
               double m,
               double n,
               double o,
               double p)
    {
      double sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
                   pow(m, 2) + pow(n, 2) + pow(o, 2) + pow(p, 2);
      if (exp(-1 * sum / (2 * pow(0.01, 2))) *
            (1 / pow(sqrt(2 * PI) * 0.01, 9)) <
          0.)
        printf("negative value:%f\n",
               exp(-1 * sum / (2 * pow(0.01, 2))) *
                 (1 / pow(sqrt(2 * PI) * 0.01, 9)));
      return exp(-1 * sum / (2 * pow(0.01, 2))) *
             (1 / pow(sqrt(2 * PI) * 0.01, 9));
    }
  };
}

int
main()
{
  float epsrel = 1.0e-3; // starting error tolerance.
  float const epsrel_min = 1.0e-9;
  float true_value = 1.;
  detail::Gauss9D integrand;

  float lows[] = {-1., -1., -1., -1., -1., -1., -1., -1., -1.};
  float highs[] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};

  constexpr int ndim = 9;
  quad::Volume<float, ndim> vol(lows, highs);

  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 0;

  PrintHeader();
  while (floatIntegrands::cu_time_and_call<detail::Gauss9D, ndim>("Gauss9D",
                                                                  integrand,
                                                                  epsrel,
                                                                  true_value,
                                                                  "gpucuhre",
                                                                  std::cout,
                                                                  configuration,
                                                                  &vol) ==
           true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }
}
