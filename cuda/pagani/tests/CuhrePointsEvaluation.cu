#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/demos/function.cuh"
#include "cuda/pagani/quad/GPUquad/Pagani.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

 class Easy {
  public:
    __device__ __host__ double
    operator()(double x,
               double y,
			   double z)
    {
      return x*y*z;
    }
  };

TEST_CASE("Compare with Cuhre")
{
	constexpr int ndim = 3;
	quad::Pagani<double, ndim> alg;
	Easy integrand;
	//double lows[] = {-1., -2., -0.5}; // original bounds
	//double highs[] = {1., 2.0, .5};
	//quad::Volume<double, ndim> vol(lows, highs);
	
	auto ResultObject = alg.EvaluateAtCuhrePoints<Easy>(integrand/*, &vol*/);
	std::cout<<ResultObject.numFuncEvals<<"\n";
	for(int i=0; i< ResultObject.numFuncEvals; ++i)
		std::cout<< ResultObject.results[i]<<"\n";
	
	std::cout<<"====\n";
	for(auto& i : ResultObject.results)
		std::cout<<i<<"\n";
}