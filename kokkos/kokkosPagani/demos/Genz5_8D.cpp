#include "kokkos/kokkosPagani/quad/Cuhre.cuh"
#include "kokkos/kokkosPagani/quad/Rule.cuh"
#include "kokkos/kokkosPagani/demos/demo_utils.cuh"
#include "kokkos/kokkosPagani/quad/func.cuh"

class GENZ_5_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
    double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta) - 10. * fabs(q - beta);
    return exp(t1);
  }
};

int
main()
{
  Kokkos::initialize();
  {
    GENZ_5_8D integrand;

    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 2.425217625641885e-06;
    const int ndim = 8;
    while (time_and_call<GENZ_5_8D, ndim>(
             "8D f5", integrand, epsrel, true_value, std::cout) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }
  Kokkos::finalize();
  return 0;
}