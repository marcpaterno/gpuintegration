#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include <array>
#include <cuda_profiler_api.h>
#include "cuda/integrands.cuh"

int main(int argc, char** argv){
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  std::array<double, 8> point_8D = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8};
  std::array<double, 7> point_7D = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7};
  std::array<double, 6> point_6D = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6};
  std::array<double, 5> point_5D = {0.1, 0.2, 0.3, 0.4 , 0.5};
  std::array<double, 4> point_4D = {0.1, 0.2, 0.3, 0.4};
  std::array<double, 3> point_3D = {0.1, 0.2, 0.3};

  double sum = 0.;
  sum += execute_integrand<Addition_8D, 8>(point_8D, num_invocations);
  sum += execute_integrand<Addition_7D, 7>(point_7D, num_invocations);
  sum += execute_integrand<Addition_6D, 6>(point_6D, num_invocations);
  sum += execute_integrand<Addition_5D, 5>(point_5D, num_invocations);
  sum += execute_integrand<Addition_4D, 4>(point_4D, num_invocations);
  sum += execute_integrand<Addition_3D, 3>(point_3D, num_invocations);
  
  return 0;
}
