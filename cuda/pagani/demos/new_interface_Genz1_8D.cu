#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_1_8D {
public:
  __device__ __host__ float
  operator()(float s,
             float t,
             float u,
             float v,
             float w,
             float x,
             float y,
             float z)
  {
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y +
               8. * z);
  }
};

int main(){
    
    float epsrel = 1.0e-3;
    float const epsrel_min = 1.0240000000000002e-10;
    constexpr int ndim = 8;
    GENZ_1_8D integrand;
    float true_value =
    (1. / 315.) * sin(1.) * sin(3. / 2.) * sin(2.) * sin(5. / 2.) * sin(3.) *
    sin(7. / 2.) * sin(4.) *
    (sin(37. / 2.) - sin(35. / 2.)); /*0.000041433844333568199264*/	
    quad::Volume<float, ndim> vol;

    while (clean_time_and_call<GENZ_1_8D, float, ndim, false>("f1",
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
	while (clean_time_and_call<GENZ_1_8D, float, ndim, true>("f1",
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

