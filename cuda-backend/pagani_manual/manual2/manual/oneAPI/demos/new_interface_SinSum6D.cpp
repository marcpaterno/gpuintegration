#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include "oneAPI/quad/Cubature_rules.h"
#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/GPUquad/Rule.h"
#include "oneAPI/quad/GPUquad/Phases.h"
#include "oneAPI/quad/Cubature_rules.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>
#include "oneAPI/quad/Workspace.h"

#include "oneapi/mkl.hpp"
#include "oneapi/tbb.h"
#include <limits>

class SinSum6D {
public:
  double
  operator()(double x, double y, double z, double k, double l, double m)const
  {
    return sin(x + y + z + k + l + m);
  }
};

class Product {
public:
  double
  operator()(double x, double y)const
  {
    return x*y;
  }
};

int
main()
{    
  sycl::queue q;  
  ShowDevice(q);  
  Product integrand;  
  constexpr size_t ndim = 2;
    
  //setup integration rules  
  //Cubature_rules<ndim> rules(q);
    
  //setup starting sub-regions
  size_t divs_per_dim = 2;
  size_t num_starting_regs = pow((double)divs_per_dim, (double)ndim);
    
  Sub_regions<ndim> regions(q, divs_per_dim);  
  double epsrel = 1.e-3;
  double epsabs = 1.e-12;

  double* lows = malloc_shared<double>(ndim, q);  
  double* highs = malloc_shared<double>(ndim, q);  
  
  for(size_t dim = 0; dim < ndim; ++dim){
    lows[dim] = 0.;
    highs[dim] = 1.;
  }
    
  
  Workspace<2> workspace(q);  
  auto  res = workspace.integrate<Product>(q, integrand, lows, highs, regions, epsrel, epsabs);
  std::cout<<"Result:"<<res.estimate << " +- " << res.errorest << std::endl;
  return 0;
}
