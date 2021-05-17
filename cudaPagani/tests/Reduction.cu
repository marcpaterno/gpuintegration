#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "demos/function.cuh"
#include "quad/GPUquad/Sample.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

class PTest {
public:   
    __device__ __host__ 
    double operator()(double x, double y){
        double res = 15.37;
        return res;
    }
};

class NTest {
public:   
    __device__ __host__ 
    double operator()(double x, double y){
        double res = -15.37;
        return res;
    }
};

class ZTest {
public:   
    __device__ __host__ 
    double operator()(double x, double y){
        return 0.;
    }
};

TEST_CASE("Constant Positive Value Function")
{
   constexpr int ndim = 2;
    size_t numRegions = 16;
    PTest integrand;
    size_t maxIters = 1;
	int heuristicID = 0; 
    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    int key = 0;
    int verbose = 0;
    int numDevices = 1;
    Cuhre<double, 2> cuhre(0, nullptr, key, verbose, numDevices);
    cuhreResult res = cuhre.integrate<PTest>(integrand, epsrel, epsabs);
        
    double integral = res.estimate;
    double error = res.errorest;
    
    //returns are never precisely equal to 0. and 15.37
	printf("ttotalEstimate:%.15f\n", integral);
    CHECK(abs(integral - 15.37) <= .00000000000001);
   
}