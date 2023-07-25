#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include <numeric>
#include <vector>
#include "common/cuda/Interp1D.cuh"
#include <iostream>
#include "common/cuda/Interp2D.cuh"
#include "common/cuda/cudaMemoryUtil.h"

// oneAPI test cannot be replicated due to no equivalent to cudaMemGetInfo

__global__ void
Evaluate(quad::Interp1D interpolator, size_t size, double* results)
{
  double val = 1.5;
  results[0] = interpolator(val);
}

template <size_t s, size_t nx, size_t ny>
class Test_object {
public:
  Test_object(double* xs_1D,
              double* ys_1D,
              std::array<double, nx> xs_2D,
              std::array<double, ny> ys_2D,
              std::array<double, ny * nx> zs_2D)
    : obj_1D(xs_1D, ys_1D, s), obj_2D(xs_2D, ys_2D, zs_2D){};

  __device__ __host__ double
  operator()()
  {
    return obj_1D(1.5) * obj_2D(2.6, 4.1);
  }

  quad::Interp1D obj_1D;
  quad::Interp2D obj_2D;
};

template <typename F>
__global__ void
Evaluate_test_obj(F* f, double* results)
{

  for (int i = 0; i < 1000; i++) {
    results[i] = f->operator()();
  }
}

template <size_t rows, size_t cols>
std::array<double, rows * cols>
create_values(std::array<double, rows> xs_2D, std::array<double, cols> ys_2D)
{
  auto fxy = [](double x, double y) { return 3 * x * y + 2 * x + 4 * y; };
  std::array<double, rows * cols> zs_2D;

  for (std::size_t i = 0; i != rows; ++i) {
    double x = xs_2D[i];

    for (std::size_t j = 0; j != cols; ++j) {
      double y = ys_2D[j];
      zs_2D[j * rows + i] = fxy(x, y);
    }
  }
  return zs_2D;
}

TEST_CASE("No memory Leak")
{

  double* results = quad::cuda_malloc<double>(1000);
  size_t pre_interp_alloc_mem = quad::get_free_mem();

  constexpr size_t s = 1000000;
  std::vector<double> xs_1D(s);
  std::vector<double> ys_1D(s);

  std::iota(xs_1D.begin(), xs_1D.end(), 1.);
  std::iota(ys_1D.begin(), ys_1D.end(), 2.);

  constexpr std::size_t nx = 3; // rows
  constexpr std::size_t ny = 2; // cols
  std::array<double, nx> xs_2D = {1., 2., 3.};
  std::array<double, ny> ys_2D = {4., 5.};
  auto zs_2D = create_values<nx, ny>(xs_2D, ys_2D);

  using IntegT = Test_object<s, nx, ny>;
  IntegT host_obj(xs_1D.data(), ys_1D.data(), xs_2D, ys_2D, zs_2D);

  size_t mem_after_one_host_object = quad::get_free_mem();
  size_t approx_obj_size = pre_interp_alloc_mem - mem_after_one_host_object;
  size_t init_object_capacity = mem_after_one_host_object / approx_obj_size;

  for (int i = 0; i < 1000; ++i) {

    IntegT* device_obj = quad::cuda_copy_to_device(host_obj);
    size_t mem_after_dev_object = quad::get_free_mem();
    ;

    Evaluate_test_obj<IntegT><<<1, 1>>>(device_obj, results);
    cudaDeviceSynchronize();

    size_t object_capacity = mem_after_dev_object / approx_obj_size;
    CHECK(object_capacity >= init_object_capacity - 1);
    cudaFree(device_obj);
  }

  cudaFree(results);
}
