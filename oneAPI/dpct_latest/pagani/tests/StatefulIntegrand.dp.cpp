#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "externals/catch2/catch.hpp"
#include "oneAPI/dpct_latest/pagani/quad/GPUquad/Pagani.dp.hpp"
#include "oneAPI/dpct_latest/pagani/quad/quad.h"
#include "oneAPI/dpct_latest/pagani/quad/util/cudaArray.dp.hpp"
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
  ToyModelWithHeapArray(int const* d, size_t s) { data.Initialize(d, s); }

  ToyModelWithHeapArray() = default;

  double
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

  double
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

  double
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

  double
  operator()(double x, double y) const
  {
    return model(x, y);
  }

  double true_value = 34.0;
  ToyModel model;
};

template <typename T>
void
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
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
  int rc = -1;
  double result = 0.0;
  IntegT* ptr_to_thing_in_unified_memory = cuda_copy_to_managed(on_host);

  double* device_result = cuda_malloc_managed<double>();
  int* device_rc = cuda_malloc_managed<int>();
    q_ct1.parallel_for(
      sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
          do_gpu_stuff<IntegT>(
            ptr_to_thing_in_unified_memory, device_result, device_rc);
      });
  dev_ct1.queues_wait_and_throw();
  sycl::free(ptr_to_thing_in_unified_memory, q_ct1);
  result = *device_result;
  sycl::free(device_result, q_ct1);
  sycl::free(device_rc, q_ct1);
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
