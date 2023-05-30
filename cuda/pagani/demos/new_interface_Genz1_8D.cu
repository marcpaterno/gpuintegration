#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "cuda/pagani/demos/compute_genz_integrals.cuh"
#include "common/cuda/integrands.cuh"

class GENZ_1_8D {
public:
  __device__ __host__ double
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

int main(){
    std::cout<<"here\n";
    double epsrel = 1.0e-3;
    double const epsrel_min = 1.0240000000000002e-10;
    constexpr int ndim = 8;
    F_1_8D integrand;
    double true_value =
    (1. / 315.) * sin(1.) * sin(3. / 2.) * sin(2.) * sin(5. / 2.) * sin(3.) *
    sin(7. / 2.) * sin(4.) *
    (sin(37. / 2.) - sin(35. / 2.)); /*0.000041433844333568199264*/	
    quad::Volume<double, ndim> vol;
	std::cout<<true_value<<std::endl;
    printf("%.15e\n", true_value);
	while (clean_time_and_call<F_1_8D, double, ndim, false>("f1",
                                           integrand,
                                           epsrel,
                                           true_value,
                                           "gpucuhre",
                                           std::cout,
										   vol) == true &&
         epsrel >= epsrel_min) {
			epsrel /= 5.0;
	}
	
    return 0;
}

