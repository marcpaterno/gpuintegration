#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/quad/GPUquad/Pagani.cuh"
#include "cuda/pagani/quad/quad.h"
#include "common/cuda//cudaArray.cuh"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

template <class M>
M
make_from_file(char const* filename)
{
  static_assert(std::is_default_constructible<M>::value,
                "Type must be default constructable");
  char const* basedir = std::getenv("PAGANI_DIR");
  std::string fname(basedir);
  fname += "/tests/";
  fname += filename;
  std::cout << "Filename:" << fname << std::endl;
  std::ifstream in(fname);
  if (!in) {
    std::string msg("Failed to open file: ");
    msg += fname;
    throw std::runtime_error(msg);
  }

  M result;
  in >> result;
  return result;
}

struct ToyModelWithHeapArray {
  ToyModelWithHeapArray(int const* d, size_t s): data(d, s) {}

  ToyModelWithHeapArray() = default;

  __host__ __device__ double
  operator()(double x, double y) const
  {
    double answer = data[0] + x * data[1] + y * data[2] + x * y * data[3];
    return answer;
  }

  gpu::cudaDynamicArray<int> data;
};

struct ToyIntegrandWithHeapModel {

  explicit ToyIntegrandWithHeapModel(int* const modeldata, size_t s)
    : model(modeldata, s)
  {}

  __host__ __device__ double
  operator()(double x, double y) const
  {
    return model(x, y);
  }

  double true_value = 34.0;
  ToyModelWithHeapArray model;
};

struct ToyModel {

  explicit ToyModel(int const* d)
  {
    // std::memcpy(data.data, d, sizeof(int) * 5);
    data.Initialize(d);
  }

  ToyModel() = default;

  __host__ __device__ double
  operator()(double x, double y) const
  {
    double answer = data[0] + x * data[1] + y * data[2] + x * y * data[3];
    return answer;
  }

  gpu::cudaArray<int, 5> data;
}; // ToyModel

struct ToyIntegrand {

  explicit ToyIntegrand(std::array<int, 5> const& modeldata)
    : model(modeldata.data())
  {}

  __host__ __device__ double
  operator()(double x, double y) const
  {
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

template <typename IntegT>
std::pair<double, int>
toy_integration_algorithm(IntegT const& on_host)
{
  int rc = -1;
  double result = 0.0;
  IntegT* ptr_to_thing_in_unified_memory = cuda_copy_to_managed(on_host);

  double* device_result = cuda_malloc_managed<double>();
  int* device_rc = cuda_malloc_managed<int>();
  do_gpu_stuff<IntegT>
    <<<1, 1>>>(ptr_to_thing_in_unified_memory, device_result, device_rc);
  cudaDeviceSynchronize();
  cudaFree(ptr_to_thing_in_unified_memory);
  result = *device_result;
  cudaFree(device_result);
  cudaFree(device_rc);
  return {result, rc};
}

TEST_CASE("State of Integrand is std::array")
{
  std::array<int, 5> attributes{5, 4, 3, 2, 1};
  ToyIntegrand my_integrand(attributes);
  auto [result, rc] = toy_integration_algorithm(my_integrand);

  CHECK(result == 34.0);
  CHECK(result == my_integrand.true_value);
}

TEST_CASE("Dynamic Array")
{
  std::array<int, 5> attributes{5, 4, 3, 2, 1};
  ToyIntegrandWithHeapModel my_integrand(attributes.data(), attributes.size());
  auto [result, rc] = toy_integration_algorithm(my_integrand);
  CHECK(result == 34.0);
  CHECK(result == my_integrand.true_value);
}
