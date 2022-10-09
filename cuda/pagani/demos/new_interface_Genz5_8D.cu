#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_5_8D {
  public:
    __device__ __host__ float
    operator()(float x,
               float y,
               float z,
               float k,
               float m,
               float n,
               float p,
               float q)
    {
      float beta = .5;
      float t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                  10. * fabs(z - beta) - 10. * fabs(k - beta) -
                  10. * fabs(m - beta) - 10. * fabs(n - beta) -
                  10. * fabs(p - beta) - 10. * fabs(q - beta);
      return exp(t1);
    }
};

int main(){
    
    float epsrel = 1.0e-3;
    float const epsrel_min = 1.0240000000000002e-10;
    constexpr int ndim = 8;
    GENZ_5_8D integrand;
	float true_value = 2.425217625641885e-06;
	quad::Volume<float, ndim>  vol;
	
	
    while (clean_time_and_call<GENZ_5_8D, float, ndim, false>("f5",
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
	while (clean_time_and_call<GENZ_5_8D, float, ndim, true>("f5",
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