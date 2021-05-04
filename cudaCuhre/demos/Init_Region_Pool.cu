#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"
#include "nvToolsExt.h" 

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

int main(){

    constexpr int ndim = 3;
    DIM3 integrand;
    quad::Cuhre<double, ndim> alg(0, nullptr);
    
    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    bool phase_2 = false;
    int phase_I_type = 0;
    int heuristicID = 0;
    int maxIters = 15;
    int outfileVerbosity = 0;
    int _final = 1;
    quad::Volume<double, ndim>* vol = nullptr;
    using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
    auto const t0 = std::chrono::high_resolution_clock::now();
    cuhreResult const result = alg.dummyintegrate(integrand, epsrel, epsabs, vol, outfileVerbosity, maxIters, _final, heuristicID, phase_I_type, phase_2);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    std::cout<<"Total time:"<< dt.count() <<"ms\n";
    return 0;    
}