#include "kokkos/pagani/demos/demo_utils.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class GENZ_1_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
  {
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y +
               8. * z);
  }
};

int
main()
{
  Kokkos::initialize();
  {
    GENZ_1_8D integrand;
    constexpr bool use_custom = true;
    constexpr int debug = 0;

    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = (1. / 315.) * sin(1.) * sin(3. / 2.) * sin(2.) *
                        sin(5. / 2.) * sin(3.) * sin(7. / 2.) * sin(4.) *
                        (sin(37. / 2.) - sin(35. / 2.));
    const int ndim = 8;
    while (time_and_call<GENZ_1_8D, ndim, use_custom, debug>(
             "f1", integrand, epsrel, true_value) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }
  Kokkos::finalize();
  return 0;
}