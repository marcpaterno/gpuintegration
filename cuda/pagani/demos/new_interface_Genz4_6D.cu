#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_4_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v, double b)
  {
    // double alpha = 25.;
    double beta = .5;
    return exp(-1.0 * pow(25, 2) *
               (pow(x - beta, 2) + pow(y - beta, 2) + pow(z - beta, 2) +
                pow(w - beta, 2) + pow(v - beta, 2) + pow(b - beta, 2)));
  }
};

int
main()
{

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 6;
  GENZ_4_6D integrand;
  double true_value = 1.79132603674879e-06;
  quad::Volume<double, ndim> vol;

  // for(int i=0; i < 10; ++i)
  call_cubature_rules<GENZ_4_6D, ndim>(integrand, vol);

  /*while (clean_time_and_call<GENZ_4_5D, double, ndim, false>("f4",
                                         integrand,
                                         epsrel,
                                         true_value,
                                         "gpucuhre",
                                         std::cout,
                                                                                 vol) == true &&
       epsrel >= epsrel_min) {
                      epsrel /= 5.0;
      }*/

  /*epsrel = 1.0e-3;
  while (clean_time_and_call<GENZ_4_5D, double, ndim, true>("f4",
                                     integrand,
                                     epsrel,
                                     true_value,
                                     "gpucuhre",
                                     std::cout) == true &&
   epsrel >= epsrel_min) {
                  epsrel /= 5.0;
  }*/
  return 0;
}
