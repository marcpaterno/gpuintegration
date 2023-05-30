#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "cuda/pagani/demos/compute_genz_integrals.cuh"
#include "common/cuda/integrands.cuh"

class GENZ_6_6D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z)
  {
    if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(10 * z + 9 * y + 8 * x + 7 * w + 6 * v + 5 * u);
  }
};

int
main()
{

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 5;
  GENZ_6_6D integrand;
  double true_value = 1.5477367885091207413e8;
  quad::Volume<double, ndim> vol;
  std::cout<<"genz-compute answer: " << compute_discontinuous<6>({5., 6., 7., 8., 9., 10.}, {.4, .5, .6, .7, .8, .9}) <<std::endl;
  F_6_5D temp;
  temp.set_true_value();
  std::cout<<"genz-compute answer for 5D:"<< temp.true_value << std::endl;
	
  while (clean_time_and_call<F_6_5D, double, ndim, false>(
           "f6", temp, epsrel, true_value, "gpucuhre", std::cout, vol) ==
           true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  /*epsrel = 1.0e-3;
  while (clean_time_and_call<GENZ_6_6D, ndim, true>("f6",
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
