#include "kokkos/pagani/demos/demo_utils.cuh"
#include "kokkos/pagani/quad/func.cuh"

class GENZ3_3D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z)
  {
    return pow(1 + 3 * x + 2 * y + z, -4);
  }
};

int
main()
{
  Kokkos::initialize();
  {
    GENZ3_3D integrand;
    constexpr bool use_custom = true;
    constexpr int debug = 0;
    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 0.010846560846560846561;
    const int ndim = 3;
    while (time_and_call<GENZ3_3D, ndim, use_custom, debug>(
             "3D f3", integrand, epsrel, true_value) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }
  Kokkos::finalize();
  return 0;
}