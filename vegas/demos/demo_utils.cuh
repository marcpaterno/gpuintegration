#ifndef VEGAS_DEMO_UTILS_CUH
#define VEGAS_DEMO_UTILS_CUH

#include <iomanip>
#include <chrono>
#include <iostream>
#include "vegas/vegasT.cuh"
#include <map>
#include <string>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

bool ApproxEqual(double a, double b, double epsilon = 1.e-5){
    if(std::abs(a-b) <= epsilon)
        return true;
    return false;
}

struct VegasParams{
    
  VegasParams(double callsPerIter, double total_iters, double adjust_iters, int skipIters): ncall(callsPerIter), t_iter(total_iters), num_adjust_iters(adjust_iters),num_skip_iters(skipIters){};
    
  double ncall = 1.e7;
  int t_iter = 70;
  int num_adjust_iters = 40;
  int num_skip_iters = 5;
    
};

void PrintHeader(){
    std::cout << "id, intgral, estimate, std, chi, iters, adj_iters, skip_iters, ncall, neval"
               "time, status\n";
}


template <typename F, int ndim>
bool
mcubes_time_and_call(F integrand,
              double epsrel,
              double correct_answer,
              char const* integralName,
              VegasParams params,
              quad::Volume<double, ndim>* volume)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-20;
  bool success = false;
  do{
      //std::cout<<"Trying with "<< params.ncall << " and "<< params.num_adjust_iters<< " adjust iters\n";
      double exp_epsrel = epsrel*.5;
      auto t0 = std::chrono::high_resolution_clock::now();
      auto res = cuda_mcubes::integrate<F, ndim>
        (integrand, ndim, exp_epsrel, epsabs, params.ncall, volume, params.t_iter, params.num_adjust_iters, params.num_skip_iters);
      MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
      success = (res.status == 0);
      std::cout.precision(15);
	  if(success)
		std::cout << integralName << "," 
            << epsrel << ","
            << std::scientific << correct_answer << "," 
            << std::scientific << res.estimate << "," 
            << std::scientific << res.errorest << "," 
            << res.chi_sq << "," 
            << params.t_iter <<","
            << params.num_adjust_iters << ","
            << params.num_skip_iters << ","
            << params.ncall <<","
            << res.neval <<","
            << dt.count() << ","
            << res.status << "\n";
      
  }while (success == false && AdjustParams(params.ncall, params.t_iter) == true);
  
  return success;
}

#endif