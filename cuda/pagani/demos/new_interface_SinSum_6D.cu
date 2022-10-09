#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class SinSum6D {
  public:
    __device__ __host__ float
    operator()(float x, float y, float z, float k, float l, float m)
  {
    return sin(x + y + z + k + l + m);
  }
};

int main(){
    
    float epsrel = 1.0e-3;
    float const epsrel_min = 1.0240000000000002e-10;
    constexpr int ndim = 6;
    SinSum6D integrand;
    float true_value = -49.165073;
	std::array<float, ndim> lows = {0., 0., 0., 0., 0., 0.};
	std::array<float, ndim> highs = {10., 10., 10., 10., 10., 10.};
	quad::Volume<float, ndim>  vol(lows, highs);
	
	
    while (clean_time_and_call<SinSum6D, float, ndim, false>("SinSum",
                                           integrand,
                                           epsrel,
                                           true_value,
                                           "gpucuhre",
                                           std::cout,
										   vol) == true &&
         epsrel >= epsrel_min) {
			epsrel /= 5.0;
	}
	
	/*epsrel = 1.0e-3;
	while (clean_time_and_call<SinSum6D, float, ndim, true>("f5",
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

