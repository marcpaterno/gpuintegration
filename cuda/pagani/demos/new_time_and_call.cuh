#ifndef ALTERNATIVE_TIME_AND_CALL_CUH
#define ALTERNATIVE_TIME_AND_CALL_CUH

#include <chrono>
#include "cuda/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"
#include "common/integration_result.hh"
#include "common/cuda/Volume.cuh"
#include "nvToolsExt.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <string>
#include <limits>
#include <stdlib.h>
#include <vector>

template <typename T>
void
host_print_dev_array(T* dev, size_t size, std::string label)
{
  T* host = new T[size];
  cuda_memcpy_to_host(host, dev, size);
  for (int i = 0; i < size; ++i)
    std::cout << label << "," << i << "," << std::scientific
              << std::setprecision(15) << host[i] << std::endl;
  delete[] host;
}

template <typename F, int ndim>
void
call_cubature_rules(F integrand,
                    quad::Volume<double, ndim>& vol,
                    int num_repeats = 11)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // cudaDeviceReset();

  std::cout << "num-repeats:" << num_repeats << std::endl;
  for (int i = 0; i < num_repeats; ++i) {
    for (int splits_per_dim = ndim >= 8 ? 5 : 8; splits_per_dim < 15;
         splits_per_dim++) {
      F* d_integrand = quad::make_gpu_integrand<F>(integrand);
      Sub_regions<double, ndim> sub_regions(splits_per_dim);
      size_t num_regions = sub_regions.size;

      if (num_regions >= 35e6 || num_regions * 64 / INT_MAX >= 1.)
        break;
      Region_characteristics<ndim> characteristics(sub_regions.size);
      Region_estimates<double, ndim> estimates(sub_regions.size);
      Cubature_rules<double, ndim> rules;

      rules.set_device_volume(vol.lows, vol.highs);
      int iteration = 0;
      bool compute_relerr_error_reduction = false;

      numint::integration_result iter =
        rules.template apply_cubature_integration_rules<F>(
          d_integrand,
          iteration,
          sub_regions,
          estimates,
          characteristics,
          compute_relerr_error_reduction);

      // sub_regions.print_bounds();
      cudaFree(d_integrand);
    }
  }
}

template <typename F, int ndim>
void
call_cubature_rules(int num_repeats = 11)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  // cudaDeviceReset();
  for (int i = 0; i < num_repeats; ++i) {
    for (int splits_per_dim = ndim >= 8 ? 5 : 8; splits_per_dim < 15;
         splits_per_dim++) {
      quad::Volume<double, ndim> vol;
      F integrand;
      F* d_integrand = quad::make_gpu_integrand<F>(integrand);
      Sub_regions<double, ndim> sub_regions(splits_per_dim);
      size_t num_regions = sub_regions.size;

      if (num_regions >= 35e6 || num_regions * 64 / INT_MAX >= 1.)
        break;
      Region_characteristics<ndim> characteristics(sub_regions.size);
      Region_estimates<double, ndim> estimates(sub_regions.size);
      Cubature_rules<double, ndim> rules;
      rules.set_device_volume(vol.lows, vol.highs);
      int iteration = 0;
      bool compute_relerr_error_reduction = false;

      numint::integration_result iter =
        rules.template apply_cubature_integration_rules<F>(
          d_integrand,
          iteration,
          sub_regions,
          estimates,
          characteristics,
          compute_relerr_error_reduction);

      // sub_regions.print_bounds();
      std::cout << "estimates:" << std::scientific << std::setprecision(15)
                << std::scientific << iter.estimate << "," << num_regions
                << std::endl;
      cudaFree(d_integrand);
    }
  }
}

template <typename F,
          typename T,
          int ndim,
          bool use_custom = false,
          int debug = 0>
bool
clean_time_and_call(std::string id,
                    F integrand,
                    T epsrel,
                    T true_value,
                    char const* algname,
                    std::ostream& outfile,
                    quad::Volume<T, ndim>& vol,
                    bool relerr_classification = true)
{

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  T constexpr epsabs = 1.0e-40;
  bool good = false;
  Workspace<T, ndim, use_custom> workspace;

  auto print_custom = [=](bool use_custom_flag) {
    std::string to_print = use_custom_flag == true ? "custom" : "library";
    return to_print;
  };

  for (int i = 0; i < 3; i++) {

    auto const t0 = std::chrono::high_resolution_clock::now();
    size_t partitions_per_axis = 2;
    if (ndim < 5)
      partitions_per_axis = 4;
    else if (ndim <= 10)
      partitions_per_axis = 2;
    else
      partitions_per_axis = 1;

    Sub_regions<T, ndim> sub_regions(partitions_per_axis);
    constexpr bool predict_split = false;
    constexpr bool collect_iters = false;

    numint::integration_result result =
      workspace.template integrate<F, predict_split, collect_iters>(
        integrand, sub_regions, epsrel, epsabs, vol, relerr_classification);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
    T const absolute_error = std::abs(result.estimate - true_value);

    if (result.status == 0) {
      good = true;
    }

    outfile.precision(17);
    if (i != 0)
      outfile << std::fixed << std::scientific << id << "," << ndim << ","
              << print_custom(use_custom) << "," << true_value << "," << epsrel
              << "," << epsabs << "," << result.estimate << ","
              << result.errorest << "," << result.nregions << ","
              << result.status << "," << dt.count() << std::endl;
  }
  return good;
}

void
print_header()
{
  std::cout << "id, ndim, use_custom, integral, epsrel, epsabs, estimate, "
               "errorest, nregions, status, time\n";
}

template <typename F, int ndim>
__global__ void
execute_integrand_kernel(F* integrand,
                         double* d_point,
                         double* output,
                         size_t num_invocations)
{
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  gpu::cudaArray<double, ndim> point;

  double start_val = .01;
#pragma unroll ndim
  for (int i = 0; i < ndim; ++i) {
    point[i] = start_val * (i + 1);
  }

  double total = 0.;
  // #pragma unroll 1
  for (int i = 0; i < num_invocations; ++i) {

    double res = gpu::apply(*integrand, point);
    total += res;
  }
  output[tid] = total;
}

template <typename F, int ndim>
double
execute_integrand(std::array<double, ndim> point, size_t num_invocations)
{
  const size_t num_blocks = 1024;
  const size_t num_threads = 64;

  F integrand;
  F* d_integrand = quad::cuda_copy_to_managed<F>(integrand);

  double* d_point = quad::cuda_malloc<double>(point.size());
  quad::cuda_memcpy_to_device(d_point, point.data(), point.size());

  double* output = quad::cuda_malloc<double>(num_threads * num_blocks);

  for (int i = 0; i < 10; ++i) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    execute_integrand_kernel<F, ndim><<<num_blocks, num_threads>>>(
      d_integrand, d_point, output, num_invocations);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start, stop);
    std::cout << "time:" << kernel_time << std::endl;
  }
  std::cout << "time---------\n";

  std::vector<double> host_output;
  host_output.resize(num_threads * num_blocks);
  // std::cout<<"vector size:"<<host_output.size()<<std::endl;
  quad::cuda_memcpy_to_host<double>(
    host_output.data(), output, host_output.size());

  double sum = 0.;
  for (int i = 0; i < num_threads * num_blocks; ++i)
    sum += host_output[i];

  cudaFree(output);
  cudaFree(d_integrand);
  cudaFree(d_point);
  return sum;
}

template <typename F, int ndim>
__global__ void
execute_integrand_kernel_at_points(F* integrand,
                                   double* points,
                                   double* output,
                                   size_t num_invocations)
{
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  gpu::cudaArray<double, ndim> point;

  double total = 0.;
#pragma unroll 1
  for (int i = 0; i < num_invocations; ++i) {

#pragma unroll ndim
    for (int dim = 0; dim < ndim; ++dim) {
      point[dim] = points[i * ndim + dim];
    }

    double res = gpu::apply(*integrand, point);
    total += res;
  }
  output[tid] = total;
}

template <typename F, int ndim>
double
execute_integrand_at_points(size_t num_invocations)
{
  const size_t num_blocks = 1024;
  const size_t num_threads = 64;

  F integrand;
  F* d_integrand = quad::cuda_copy_to_managed<F>(integrand);

  double* output = quad::cuda_malloc<double>(num_threads * num_blocks);
  double* points = quad::cuda_malloc<double>(num_invocations * ndim);
  std::vector<double> h_points;
  srand(1);

  for (int i = 0; i < num_invocations * ndim; ++i)
    h_points.push_back(rand());

  quad::cuda_memcpy_to_device<double>(points, h_points.data(), h_points.size());

  for (int i = 0; i < 10; ++i) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    execute_integrand_kernel_at_points<F, ndim><<<num_blocks, num_threads>>>(
      d_integrand, points, output, num_invocations);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, start, stop);
    std::cout << "time:" << kernel_time << std::endl;
  }
  std::cout << "time---------\n";

  std::vector<double> host_output;
  host_output.resize(num_threads * num_blocks);
  // std::cout<<"vector size:"<<host_output.size()<<std::endl;
  quad::cuda_memcpy_to_host<double>(
    host_output.data(), output, host_output.size());

  double sum = 0.;
  for (int i = 0; i < num_threads * num_blocks; ++i)
    sum += host_output[i];

  cudaFree(output);
  cudaFree(d_integrand);
  cudaFree(points);
  return sum;
}

#endif
