#include "kokkos/kokkosPagani/quad/Cuhre.cuh"
#include "kokkos/kokkosPagani/quad/Rule.cuh"
#include "kokkos/kokkosPagani/demos/demo_utils.cuh"
#include "kokkos/kokkosPagani/quad/func.cuh"

class GENZ3_3D {
public:
  __device__ __host__ double
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

    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 0.010846560846560846561;
    const int ndim = 3;
    while (time_and_call<GENZ3_3D, ndim>(
             "3D f3", integrand, epsrel, true_value, std::cout) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }
  Kokkos::finalize();
  return 0;
}