#include "kokkos/pagani/quad/GPUquad/Sub_regions.cuh"
//#include "kokkos/pagani/quad/GPUquad/Rule.cuh"
//#include "kokkos/pagani/demos/demo_utils.cuh"
#include "func.cuh"

class BoxIntegral8_15 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {

    double s = 15;
    double sum = 0;
    sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
          pow(m, 2) + pow(n, 2) + pow(o, 2);
    return pow(sum, s / 2);
  }
};

int
main()
{
  Kokkos::initialize();
  {
    BoxIntegral8_15 integrand;
    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 8879.851175413485;
    const int ndim = 8;
	Sub_regions<double, ndim> regs(2);
	//regs.print_bounds();
	std::cout<<"total vol:"<<regs.compute_total_volume();
  }
  Kokkos::finalize();
  return 0;
}