#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "cuda/integrands.cuh"

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  constexpr int ndim = 5;
  Addition_5D integrand;
  quad::Volume<double, ndim> vol;
  call_cubature_rules<Addition_5D, ndim>(integrand, vol, num_repeats);
  return 0;
}
