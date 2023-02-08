#include "kokkos/pagani/demos/demo_utils.cuh"
#include "kokkos/pagani/quad/func.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class GENZ_4_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v)
  {
    // double alpha = 25.;
    double beta = .5;
    return exp(-1.0 *
               (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2)));
  }
};

int
main()
{
  Kokkos::initialize();
  {
    GENZ_4_5D integrand;
	constexpr bool use_custom = true;
	constexpr int debug = 0;
    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 1.79132603674879e-06;
    const int ndim = 5;
    while (time_and_call<GENZ_4_5D, ndim, use_custom, debug>(
             "5D f4", integrand, epsrel, true_value) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
      break;
    }
  }
  Kokkos::finalize();
  return 0;
}