#include "time_and_call.h"

class SinSum6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};


int
main(int argc, char** argv)
{
  Kokkos::initialize();
  {
  double epsrel = 1.e-3;
  double epsabs = 1.e-20;

  constexpr int ndim = 6;
  double ncall = 2.0e9;
  int titer = 10;
  int itmax = 0;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);

  double true_value = -49.165073;
  std::cout << "id, estimate, std, chi, iters, adj_iters, skip_iters, ncall, "
               "time, abserr, relerr\n";

  double lows[] = {0., 0., 0., 0., 0., 0.};
  double highs[] = {10., 10., 10., 10., 10., 10.};
  Volume<double, ndim> volume(lows, highs);
  SinSum6D integrand;
  
  //mcubes_time_and_call<GENZ_4_5D, ndim>(integrand, epsrel, true_value, "f4 5D", params, &volume);
  //std::array<double, 6> required_ncall = {1.e7, 1.e7, 1.e7, 1.e9, 1.e9, 8.e9};
  
  
  PrintHeader();
  //size_t expID = 0;
  bool success = false;
  success = mcubes_time_and_call<SinSum6D, ndim>(integrand, epsrel, true_value, "f4 5D", params, &volume);  
  }
  Kokkos::finalize();
  return 0;
}

