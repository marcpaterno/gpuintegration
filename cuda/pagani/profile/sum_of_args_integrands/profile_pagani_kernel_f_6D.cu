#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "common/cuda/integrands.cuh"

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 100;
  constexpr int ndim = 6;
  Addition_6D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<Addition_6D, ndim>(integrand, vol, num_repeats);
  return 0;
}
