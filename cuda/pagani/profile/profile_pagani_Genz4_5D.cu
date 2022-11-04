#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_4_5D {
  public:
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v)
    {
		double beta = .5;
		
		/*return  exp(-1.*(pow(25.,2.) * pow(x - beta, 2.) + 
				pow(25.,2.) * pow(y - beta, 2.) +
                pow(25.,2.) * pow(z - beta, 2.) + 
				pow(25.,2.) * pow(w - beta, 2.) +
                pow(25.,2.) * pow(v - beta, 2.))); 
				
				
		  
		 
		 
		return  (pow(25.,2) * pow(x - beta, 2.) + 
				pow(25.,2) * pow(y - beta, 2.) +
                pow(25.,2) * pow(z - beta, 2.) + 
				pow(25.,2) * pow(w - beta, 2.) +
                pow(25.,2) * pow(v - beta, 2.));  */
		 
		 
		/*return  25.*25. * pow(x - beta, 2.) + 
				25.*25. * pow(y - beta, 2.) +
                25.*25. * pow(z - beta, 2.) + 
				25.*25. * pow(w - beta, 2.) +
                25.*25. * pow(v - beta, 2.); 
		 
		 
		return  25.*25. * (x - beta)*(x - beta) + 
				25.*25. * (y - beta)*(y - beta) +
                25.*25. * (z - beta)*(z - beta) + 
				25.*25. * (w - beta)*(w - beta) +
                25.*25. * (v - beta)*(v - beta);
		*/
	   
      // double alpha = 25.;

      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.)));
    }
};

int
main()
{
  constexpr int ndim = 5;
  GENZ_4_5D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<GENZ_4_5D, ndim>(integrand, vol);
  return 0;
}
