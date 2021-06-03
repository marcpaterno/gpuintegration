#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../quad/util/cudaArray.cuh"
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"

using namespace quad;

struct ToyModel {

  explicit
  ToyModel(int const* d) { 
    std::memcpy(data.data, d, sizeof(int) * 5);
  }

  ToyModel() = default;

  __host__ __device__ double
  operator()(double x, double y) const
  {
    double answer = data[0] + x * data[1] + y * data[2] + x*y*data[3];
    return answer;
  }

  gpu::cudaArray<int, 5>data;
}; // ToyModel

struct ToyIntegrand {

  explicit
  ToyIntegrand(std::array<int, 5> const&  modeldata) :
     model(modeldata.data())
{ }
  
  __host__ __device__ double
  operator()(double x, double y) const {
    return model(x, y);
  }
  
  double true_value = 34.0;
  ToyModel model;
};

template <typename T>
__global__ void
do_gpu_stuff(T* t, double* integration_result, int* integration_status)
{
  double r = (*t)(2., 3.);
  *integration_result = r;
  *integration_status = 0; // Success !	
}

template<typename IntegT>
std::pair<double, int>
toy_integration_algorithm(IntegT const& on_host)
{
  int rc = -1;
  double result = 0.0;
  IntegT* ptr_to_thing_in_unified_memory = cuda_copy_to_managed(on_host);

  double* device_result = cuda_malloc_managed<double>();
  int* device_rc = cuda_malloc_managed<int>();  
  do_gpu_stuff<IntegT><<<1, 1>>>(ptr_to_thing_in_unified_memory, device_result, device_rc);
  cudaDeviceSynchronize();
  cudaFree(ptr_to_thing_in_unified_memory);
  result = *device_result;
  cudaFree(device_result);
  cudaFree(device_rc);
  return {result, rc};
}

TEST_CASE("State of Integrand is std::array"){
    std::array<int, 5> attributes { 5, 4, 3, 2, 1};
    ToyModel my_model(attributes.data());
    ToyIntegrand my_integrand(attributes);
    auto [result, rc]  = toy_integration_algorithm(my_integrand);
    
    CHECK(result == 34.0);
    CHECK(result == my_integrand.true_value);
}
