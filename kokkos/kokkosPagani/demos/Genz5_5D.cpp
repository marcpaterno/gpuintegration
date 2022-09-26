#include "kokkos/kokkosPagani/quad/Cuhre.cuh"
#include "kokkos/kokkosPagani/quad/Rule.cuh"
#include "kokkos/kokkosPagani/demos/demo_utils.cuh"
#include "kokkos/kokkosPagani/quad/func.cuh"

class GENZ_5_5D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double m)
  {
    double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta);
    return exp(t1);
  }
};

int
main()
{
  Kokkos::initialize();
  {
    GENZ_5_5D integrand;

    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 0.0003093636;
    const int ndim = 5;
    while (time_and_call<GENZ_5_5D, ndim>(
             "5D f5", integrand, epsrel, true_value, std::cout) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }
  Kokkos::finalize();
  return 0;
}