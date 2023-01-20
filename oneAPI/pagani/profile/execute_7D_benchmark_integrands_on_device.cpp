#include <CL/sycl.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "common/oneAPI/integrands.hpp"

int main(int argc, char** argv){
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  constexpr int ndim = 7;
  std::array<double, ndim> point = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7};
  double sum = 0.;
  sum += execute_integrand_at_points<F_1_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_2_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_3_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_4_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_5_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_6_7D, ndim>(num_invocations);
  printf("%.15e\n", sum);
  
  return 0;
}

