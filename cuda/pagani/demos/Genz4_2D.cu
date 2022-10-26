#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_4_2D {
  public:
    __device__ __host__ double
    operator()(double x, double y)
    {
      // double alpha = 25.;
      double beta = .5;
      return exp(
        -1.0 * (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2)));
    }
};



int main(){
    
    double epsrel = 1.0e-3;
    double const epsrel_min = 1.0240000000000002e-10;
    constexpr int ndim = 2;
    GENZ_4_2D integrand;
    double true_value = 1.79132603674879e-06;
	quad::Volume<double, ndim> vol;
	constexpr bool use_custom = false;
	constexpr int debug = 1;
	bool relerr_classification = true;
	int num_runs = 2;
	
	std::cout<<"OVER ALL EPSRELS\n";
	
    while (clean_time_and_call<GENZ_4_2D, double, ndim, use_custom>("f4",
                                           integrand,
                                           epsrel,
                                           true_value,
                                           "gpucuhre",
                                           std::cout,
										   vol, 
										   relerr_classification, 
										   num_runs) == true &&
         epsrel >= epsrel_min) {
			epsrel /= 5.0;
	}
	
	std::cout<<"========================================\n";
	
	std::cout<<"DETAILED ITER RESULTS\n";
	const double this_epsrel = 2.04800000000000032e-11;
	clean_time_and_call<GENZ_4_2D, double, ndim, use_custom, debug>("f4",
                                           integrand,
                                           this_epsrel,
                                           true_value,
                                           "gpucuhre",
                                           std::cout,
										   vol, 
										   relerr_classification);
    
	
    return 0;
}

