#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"

class Simple_3_3D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z)
  {
	double sum = 0.;
	for(int i=0; i < 1000; ++i)
		sum += (x*y*z)/(x/y/z);
	return sum;
  }
};


int
main(int argc, char** argv)
{
  Kokkos::initialize();	
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 3;
  Simple_3_3D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<Simple_3_3D, ndim>(integrand, vol, num_repeats);
  Kokkos::finalize();
  return 0;
}
