#include "cuda/cudaPagani/demos/demo_utils.cuh"
#include "cuda/cudaPagani/demos/function.cuh"
#include "cuda/cudaPagani/demos/compute_genz_integrals.cuh"
#include "cuda/mcubes/demos/demo_utils.cuh"


#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
using namespace quad;

namespace detail {
  class Gaussian {
  public:
    gpu::cudaArray<double, 5> alphas;
    gpu::cudaArray<double, 5> betas;
    
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v)
    {
      return exp(
        -1.0 * (pow(alphas[0], 2) * pow(x - betas[0], 2) + 
                pow(alphas[1], 2) * pow(y - betas[1], 2) +
                pow(alphas[2], 2) * pow(z - betas[2], 2) + 
                pow(alphas[3], 2) * pow(w - betas[3], 2) +
                pow(alphas[4], 2) * pow(v - betas[4], 2)));
    }
  };
  
  class Product_peak {
    public:
      __device__ __host__ double
      operator()(double x, double y, double z, double k, double l, double m)
      {
        double a = 50.;
        double b = .5;

        double term_1 = 1. / ((1. / pow(a, 2)) + pow(x - b, 2));
        double term_2 = 1. / ((1. / pow(a, 2)) + pow(y - b, 2));
        double term_3 = 1. / ((1. / pow(a, 2)) + pow(z - b, 2));
        double term_4 = 1. / ((1. / pow(a, 2)) + pow(k - b, 2));
        double term_5 = 1. / ((1. / pow(a, 2)) + pow(l - b, 2));
        double term_6 = 1. / ((1. / pow(a, 2)) + pow(m - b, 2));

        double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
        return val /*/((1.286889807581113e+13))*/;
      }
};
  
  
  
}

double random(double low, double high){
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(low, high);
    return distr(eng);
}

void integrate_gaussians(){
    constexpr int ndim = 5;
    detail::Gaussian integrand;
    //double epsrel = 1.e-3;
    //original values
    std::array<double, ndim> alphas = {25., 25., 25., 25., 25.};//alpha is difficulty param
    std::array<double, ndim> betas = {.5, .5, .5, .5, .5};
    Config configuration;
    quad::Volume<double, ndim> vol;
    
    std::cout<<"id, alg, difficulty, epsrel, epsabs, integral, estimate, errorest, time, status\n";
    
    for(double epsrel = 1.e-3; epsrel > 1.e-9; epsrel /= 5.){
    
        for(double difficulty = 0.; difficulty < 50; difficulty += 1.){
            for(int dim = 0; dim < 5; dim++){
                alphas[dim] = 15.;//random(0., 1.);
                betas[dim] = random(0., 1.);
                integrand.alphas[dim]= alphas[dim];
                integrand.betas[dim]= betas[dim];
            }
            
            
            double true_value = compute_gaussian<5>(alphas, betas);
            common_header_pagani_time_and_call<detail::Gaussian, ndim>("pagani", "f4 5D", integrand,
                                                       epsrel,
                                                       true_value,
                                                       difficulty,
                                                       "gpucuhre",
                                                       std::cout,
                                                       configuration);
            double ncall = 1.0e6;
            int titer = 100;
            int itmax = 20;
            int skip = 5;
            VegasParams params(ncall, titer, itmax, skip);    
            common_header_mcubes_time_and_call<detail::Gaussian, ndim>(integrand, epsrel, true_value, difficulty , "mcubes", "f4 5D", params, &vol);                                           
        }
    }
}
/*
void integrate_product_peaks(){
    constexpr int ndim = 6;
    Product_peak integrand;
    
    std::array<double, ndim> alphas = {25., 25., 25., 25., 25.};//alpha is difficulty param
    std::array<double, ndim> betas = {.5, .5, .5, .5, .5};
}*/

int
main()
{
    
   integrate_gaussians();
}
