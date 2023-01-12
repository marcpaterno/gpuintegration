#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/mcubes/demo_utils.dp.hpp"
#include "oneAPI/mcubes/vegasT.dp.hpp"


class GENZ_4_5D {
  public:
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double w, double v)
    {
      double beta = .5;
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.)));
    }
};

int
main(int argc, char** argv)
{
  double epsrel = 1e-3;
  double epsrel_min = 1.e-9;
  constexpr int ndim = 5;

  double ncall = 1.0e6;
  int titer = 100;
  int itmax = 20;
  int skip = 5;
  VegasParams params(ncall, titer, itmax, skip);

  double true_value = 1.79132603674879e-06;

  double lows[] = {0., 0., 0., 0., 0.};
  double highs[] = {1., 1., 1., 1., 1.};
  quad::Volume<double, ndim> volume(lows, highs);
  GENZ_4_5D integrand;
        
 bool success = false;  
 size_t num_epsrels = 10;
 size_t curr_epsrel = 0;
 
 std::array<double, 6> required_ncall =
   {1.e6, 1.e6, 1.e6, 1.e7, 1.e9, 8.e9};
   
   
  do {
    params.ncall = required_ncall[curr_epsrel];
    for (int run = 0; run < 100; run++) {
      success = mcubes_time_and_call<GENZ_4_5D, ndim>(
        integrand, epsrel, true_value, "f4, 5", params, &volume);
      if (!success)
        break;
    }
    epsrel /= 5.;
	curr_epsrel++;
	if(curr_epsrel > required_ncall.size())
		break;
  } while (epsrel >= epsrel_min && success == true);


  return 0;
}
