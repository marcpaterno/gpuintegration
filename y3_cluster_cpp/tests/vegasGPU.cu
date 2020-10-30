#include "catch2/catch.hpp"

#include "modules/sigma_miscent_y1_scalarintegrand.hh"
#include "../cudaCuhre/quad/util/cudaArray.cuh"

#include <iostream>
#include <chrono>					
#include "utils/str_to_doubles.hh"  
#include <vector> 					

#include <fstream>
#include <stdexcept>
#include <string>
#include <array>

//GPU integrator headers
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"

#include "RZU.cuh"
#include "sig_miscent.cuh" 
#include <limits>
	
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

class GENZ_2_2D {
public:
 __device__ __host__ double
  operator()(double x, double y)
  {
    double a = 50.;
    double b = .5;
    
    double term_1 = 1./((1./pow(a,2)) + pow(x- b, 2));
    double term_2 = 1./((1./pow(a,2)) + pow(y- b, 2));
    
    double val  = term_1 * term_2;
    return val;
  }
};

int
main()
{
  double const radius_ = 0x1p+0;
  double const zo_low_ = 0x1.999999999999ap-3;
  double const zo_high_ = 0x1.6666666666666p-2;
	
  integral<GPU> d_integrand;
  constexpr int ndim = 7;
  d_integrand.set_grid_point({zo_low_, zo_high_, radius_});
  //GENZ_2_2D d_integrand;
  
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  auto const t0 = std::chrono::high_resolution_clock::now();
  VegasGPU <integral<GPU>, ndim>(d_integrand);  
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  std::cout<<"Time:"<< dt.count() <<"\n"; 
  //VegasGPU <GENZ_2_2D, ndim>(d_integrand);								     //from 0-.5 and 0-.75 the right answer is 11564.5005525392916752025
  //from 0-.5 and 0-.5 the right answer is  5858.50661482437226368347;
  return 0;
}
