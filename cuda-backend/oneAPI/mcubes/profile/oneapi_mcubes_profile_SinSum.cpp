//#include <CL/sycl.hpp>
//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "oneAPI/mcubes/demo_utils.dp.hpp"
#include "oneAPI/mcubes/vegasT.dp.hpp"
#include "oneapi/integrands.hpp"
/*
class SinSum3D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z)
  {
    return sin(x + y + z);
  }
};


class SinSum4D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k)
  {
    return sin(x + y + z + k);
  }
};

class SinSum5D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k, double l)
  {
    return sin(x + y + z + k + l);
  }
};

class SinSum6D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

class SinSum7D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m, double n)
  {
    return sin(x + y + z + k + l + m + n);
  }
};

class SinSum8D {
public:
SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k, double l, double m, double n, double p)
  {
    return sin(x + y + z + k + l + m + n + p);
  }
};
*/
int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 100;
  double epsrel = 1e-3;
  double epsrel_min = 1.e-9;
  double ncall = 1.0e8;
  int titer = 1;
  int itmax = 1;
  int skip = 0;
  VegasParams params(ncall, titer, itmax, skip);
  double true_value = 0.010846560846560846561;
  
  quad::Volume<double, 3> volume3;
  quad::Volume<double, 4> volume4;
  quad::Volume<double, 5> volume5;
  quad::Volume<double, 6> volume6;
  quad::Volume<double, 7> volume7;
  quad::Volume<double, 8> volume8;
  SinSum3D sinsum3D;
  SinSum4D sinsum4D;
  SinSum5D sinsum5D;
  SinSum6D sinsum6D;
  SinSum7D sinsum7D;
  SinSum8D sinsum8D;
  std::array<double, 4> required_ncall = {1.e8, 1.e9, 2.e9, 3.e9};
   
  bool success = false;  
  size_t num_epsrels = 10;
  size_t run = 0;
  
  for(auto num_samples : required_ncall){
    params.ncall = num_samples;
    
	signle_invocation_time_and_call<SinSum3D, 3>(sinsum3D, epsrel, true_value, "SinSum, 3", params, &volume3, num_repeats);
	signle_invocation_time_and_call<SinSum4D, 4>(sinsum4D, epsrel, true_value, "SinSum, 4", params, &volume4, num_repeats);
	signle_invocation_time_and_call<SinSum5D, 5>(sinsum5D, epsrel, true_value, "SinSum, 5", params, &volume5, num_repeats);
	signle_invocation_time_and_call<SinSum6D, 6>(sinsum6D, epsrel, true_value, "SinSum, 6", params, &volume6, num_repeats);
	signle_invocation_time_and_call<SinSum7D, 7>(sinsum7D, epsrel, true_value, "SinSum, 7", params, &volume7, num_repeats);
	signle_invocation_time_and_call<SinSum8D, 8>(sinsum8D, epsrel, true_value, "SinSum, 8", params, &volume8, num_repeats);

	run++;
	//if(run > required_ncall.size())
	//	break;
  }

  return 0;
}
