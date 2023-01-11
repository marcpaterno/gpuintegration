//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "oneAPI/integrands.hpp"

int main(int argc, char** argv){
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  std::array<double, 8> point = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8};
  double sum = 0.;
  sum += execute_integrand_at_points<F_1_8D, 8>(num_invocations);
  sum += execute_integrand_at_points<F_2_8D, 8>(num_invocations);
  sum += execute_integrand_at_points<F_3_8D, 8>(num_invocations);
  sum += execute_integrand_at_points<F_4_8D, 8>(num_invocations);
  sum += execute_integrand_at_points<F_5_8D, 8>(num_invocations);
  sum += execute_integrand_at_points<F_6_8D, 8>(num_invocations);
  printf("%.15e\n", sum);
  
  return 0;
}

