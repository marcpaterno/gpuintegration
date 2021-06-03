#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "func.cuh"
#include "Cuhre.cuh"
#include "Rule.cuh"
#include "demo_utils.cuh"

class GENZ_4_8D {
    public:
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v, double k, double m, double n){
        //double alpha = 25.;
        double beta = .5;
        return exp(-1.0*(pow(25,2)*pow(x-beta, 2) + 
                             pow(25,2)*pow(y-beta, 2) +
                             pow(25,2)*pow(z-beta, 2) +
                             pow(25,2)*pow(w-beta, 2) +
                             pow(25,2)*pow(v-beta, 2) +
                             pow(25,2)*pow(k-beta, 2) +
                             pow(25,2)*pow(m-beta, 2) +
                             pow(25,2)*pow(n-beta, 2)));
    }
};

int main(int argc, char **argv)
{   
    Kokkos::initialize(); 
    {
        GENZ_4_8D integrand;
     
		double epsrel = 1.0e-3;
		//double epsabs = 1.0e-12;
		double epsrel_min = 1.0e-10;
		double true_value = 6.383802190004379e-10;
        const int ndim = 8;
        while (time_and_call<GENZ_4_8D, ndim>("GENZ_4_8D", integrand,
            epsrel, true_value, std::cout) == true && epsrel >= epsrel_min) {
            epsrel /= 5.0;
        }
    }
    Kokkos::finalize();  
	return 0;
}