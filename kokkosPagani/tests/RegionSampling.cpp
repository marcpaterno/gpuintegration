#define CATCH_CONFIG_RUNNER
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "catch.hpp"
#include "Cuhre.cuh"


class PTest {
public:   
    __device__ __host__ 
    double operator()(double x, double y){
        return 15.37;
    }
};

class NTest {
public:   
    __device__ __host__ 
    double operator()(double x, double y){
        return -15.37;
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
    double hRegsIntegral[16] = {0.};
    double hRegsError[16] = {0.};
    double hRegs[16*ndim] = {0.};
    double hRegsLength[16*ndim] = {0.};
    
    size_t numRegions = 16;
    PTest integrand;
    size_t maxIters = 1;    
    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    Cuhre<double, ndim> cuhre(hRegsIntegral, hRegsError, hRegs, hRegsLength, numRegions);
    cuhreResult res = cuhre.Integrate<PTest>(integrand, epsrel, epsabs, maxIters);
        
    double firstEstimate = hRegsIntegral[0];
    double totalEstimate = firstEstimate;
    double totalErrorEst = 0.;
    bool nonZeroErrFound = false;
    bool diffIntegralFound = false;
        
    SECTION("Sub-Regions Have the same Integral Estimate")
    {
        for(int regID = 1; regID < numRegions; regID++){
            diffIntegralFound = hRegsIntegral[regID] == firstEstimate ? false : true;
            nonZeroErrFound = hRegsError[regID] >= 0.00000000000001 ? true : false;
            totalEstimate += hRegsIntegral[regID];
            totalErrorEst += hRegsError[regID];
        }
            
        CHECK(diffIntegralFound == false);
    }
    
    //returns are never precisely equal to 0. and 15.37
	//printf("ttotalEstimate:%.15f\n", totalEstimate);
    CHECK(abs(totalEstimate - 15.37) <= .00000000000001);
    CHECK(nonZeroErrFound == false); 
    CHECK(totalErrorEst <= 0.00000000000001); 
}

TEST_CASE("Constant Negative Value Function")
{
    constexpr int ndim = 2;
    double hRegsIntegral[16] = {0.};
    double hRegsError[16] = {0.};
    double hRegs[16*ndim] = {0.};
    double hRegsLength[16*ndim] = {0.};    
    size_t numRegions = 16;
    NTest integrand;
    size_t maxIters = 1;
	
    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    Cuhre<double, 2> cuhre(hRegsIntegral, hRegsError, hRegs, hRegsLength, numRegions);
    cuhreResult res = cuhre.Integrate<NTest>(integrand, epsrel, epsabs, maxIters);
        
    double firstEstimate = hRegsIntegral[0];
    double totalEstimate = firstEstimate;
    double totalErrorEst = 0.;
    bool nonZeroErrFound = false;
    bool diffIntegralFound = false;
        
    SECTION("Sub-Regions Have the same Integral Estimate")
    {
        for(int regID = 1; regID < numRegions; regID++){
            diffIntegralFound = hRegsIntegral[regID] == firstEstimate ? false : true;
            nonZeroErrFound = hRegsError[regID] >= 0.00000000000001 ? true : false;
            totalEstimate += hRegsIntegral[regID];
            totalErrorEst += hRegsError[regID];
        }
            
        CHECK(diffIntegralFound == false);
    }
    
    //returns are never precisely equal to 0. and -15.37
	//printf("totalEstimate:%.15f\n", totalEstimate);
    CHECK(abs(totalEstimate - (-15.37)) <= .00000000000001);
    CHECK(nonZeroErrFound == false); 
    CHECK(totalErrorEst <= 0.00000000000001); 
}

TEST_CASE("Constant Zero Value Function")
{
    constexpr int ndim = 2;
    double hRegsIntegral[16] = {0.};
    double hRegsError[16] = {0.};
    double hRegs[16*ndim] = {0.};
    double hRegsLength[16*ndim] = {0.};    
    size_t numRegions = 16;
    ZTest integrand;
    size_t maxIters = 1;    
    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    Cuhre<double, 2> cuhre(hRegsIntegral, hRegsError, hRegs, hRegsLength, numRegions);
    cuhreResult res = cuhre.Integrate<ZTest>(integrand, epsrel, epsabs, maxIters);
        
    double firstEstimate = hRegsIntegral[0];
    double totalEstimate = firstEstimate;
    double totalErrorEst = 0.;
    bool nonZeroErrFound = false;
    bool diffIntegralFound = false;
        
    SECTION("Sub-Regions Have the same Integral Estimate")
    {
        for(int regID = 1; regID < numRegions; regID++){
            diffIntegralFound = hRegsIntegral[regID] == firstEstimate ? false : true;
            nonZeroErrFound = hRegsError[regID] >= 0.00000000000001 ? true : false;
            totalEstimate += hRegsIntegral[regID];
            totalErrorEst += hRegsError[regID];
        }
            
        CHECK(diffIntegralFound == false);
    }
    
	//printf("totalEstimate:%.15f\n", totalEstimate);
    CHECK(totalEstimate== 0.0);
    CHECK(nonZeroErrFound == false); 
    CHECK(totalErrorEst <= 0.00000000000001); 
}

int main( int argc, char* argv[] ) {
  int result = 0;
  Kokkos::initialize();
  {  
    result = Catch::Session().run( argc, argv );
  }
  Kokkos::finalize();  
  return result;
}