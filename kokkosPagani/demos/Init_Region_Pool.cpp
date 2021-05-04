#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "func.cuh"
#include "Cuhre.cuh"
#include "Rule.cuh"

class DIM8 {
    public:
    __device__ __host__ double
    operator()(double x,
                double y,
                double z,
                double k,
                double l,
                double m,
                double n,
                double o){
       return 1.;
    }
};

class DIM7 {
    public:
    __device__ __host__ double
    operator()(double x,
                double y,
                double z,
                double k,
                double l,
                double m,
                double n){
       return 1.;
    }
};

class DIM6 {
    public:
        __device__ __host__ double
        operator()(double x, double y, double z, double w, double v, double b){
           return 1.;
        }
};

class DIM5 {
    public:
        __device__ __host__ double
        operator()(double x, double y, double z, double w, double v){
           return 1.;
        }
};

class DIM4 {
    public:
        __device__ __host__ double
        operator()(double x, double y, double z, double w){
           return 1.;
        }
};

class DIM3 {
    public:
        __device__ __host__ double
        operator()(double x, double y, double z){
           return 1.;
        }
};

int main(int argc, char **argv)
{   
    Kokkos::initialize(); 
    {
		DIM3 integrand;
        const int ndim = 3;
		double epsrel = 1.0e-3;
		double epsabs = 1.0e-12;
        int heuristicID = 0;
        
        int maxIters = 15;
        
        Cuhre<double, ndim> pagani;
         using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
        auto const t0 = std::chrono::high_resolution_clock::now();
        cuhreResult const result = pagani.DummyIntegrate(integrand, epsrel, epsabs, heuristicID, maxIters);		 
        MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
        std::cout<<"Total time:"<< dt.count() <<"ms\n";

    }
    Kokkos::finalize();  
	return 0;
}