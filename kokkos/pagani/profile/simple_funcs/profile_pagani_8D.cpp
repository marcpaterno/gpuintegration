#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"

class Simple_5_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
	  double sum = 0.;
	for(int i=0; i < 1000; ++i)
		sum += (x*y*z*k*m*n*p*q)/(x/y/z/k/m/n/p/q);
	return sum;
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();	
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 8;
  Simple_5_8D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<Simple_5_8D, ndim>(integrand, vol, num_repeats);
  Kokkos::finalize();
  return 0;
}
