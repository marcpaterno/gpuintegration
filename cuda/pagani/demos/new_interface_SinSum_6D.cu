#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class SinSum6D {
  public:
    __device__ __host__ double
    operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

int main(){
    
    double epsrel = 1.0e-3;
    double const epsrel_min = 1.0240000000000002e-10;
    constexpr int ndim = 6;
    SinSum6D integrand;
    double true_value = -49.165073;
	std::array<double, ndim> lows = {0., 0., 0., 0., 0., 0.};
	std::array<double, ndim> highs = {10., 10., 10., 10., 10., 10.};
	quad::Volume<double, ndim>  vol(lows, highs);
	
	
    while (clean_time_and_call<SinSum6D, double, ndim, false>("SinSum",
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

