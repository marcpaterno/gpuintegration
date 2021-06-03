#include "func.cuh"
#include "Cuhre.cuh"
#include "Rule.cuh"
#include "demo_utils.cuh"

class GENZ_6_2D {
public:
  __device__ __host__ double
  operator()(double y, double z)
  {
	  if(z > .9 || y > .8 )
		  return 0.;
	  else
		  return exp(10*z + 9*y);
  }
};

int main(int argc, char **argv)
{   
    Kokkos::initialize(); 
    {
        GENZ_6_2D integrand;
     
		double epsrel = 1.0e-3;
		//double epsabs = 1.0e-12;
		double epsrel_min = 1.0e-10;
		double true_value = 1495369.283757217694;
        const int ndim = 2;
        while (time_and_call<GENZ_6_2D, ndim>("GENZ_6_2D", integrand,
            epsrel, true_value, std::cout) == true && epsrel >= epsrel_min) {
            epsrel /= 5.0;
        }
    }
    Kokkos::finalize();  
	return 0;
}