#include <CL/sycl.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "common/oneAPI/integrands.hpp"


int main(int argc, char** argv){
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  constexpr int ndim = 5;
  std::array<double, ndim> point = {0.1, 0.2, 0.3, 0.4 , 0.5};
  double sum = 0.;
  sum += execute_integrand_at_points<F_1_5D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_2_5D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_3_5D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_4_5D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_5_5D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_6_5D, ndim>(num_invocations);
  printf("%.15e\n", sum);
  
    return 0;
}

