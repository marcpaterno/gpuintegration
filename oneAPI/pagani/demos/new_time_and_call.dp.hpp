#ifndef ALTERNATIVE_TIME_AND_CALL_CUH
#define ALTERNATIVE_TIME_AND_CALL_CUH

#include <CL/sycl.hpp>
#include <chrono>
#include "oneAPI/pagani/quad/GPUquad/PaganiUtils.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Workspace.dp.hpp"
#include "common/oneAPI/cuhreResult.dp.hpp"
#include "common/oneAPI/Volume.dp.hpp"
#include <iostream>
#include <iomanip>
#include "common/oneAPI/cudaMemoryUtil.h"
#include <limits>
#include <stdlib.h>

template <typename T>
void
host_print_dev_array(T* dev, size_t size, std::string label)
{
  T* host = new T[size];
  quad::cuda_memcpy_to_host(host, dev, size);
  for (int i = 0; i < size; ++i)
    std::cout << label << "," << i << "," << std::scientific
              << std::setprecision(15) << host[i] << std::endl;
  printf("done\n");
  delete[] host;
}

template <typename F, int ndim>
void
call_cubature_rules(int num_repeats = 11)
{
  F integrand;
  quad::Volume<double, ndim> vol;
  F* d_integrand = quad::make_gpu_integrand<F>(integrand);

  for (int i = 0; i < num_repeats; ++i) {
    for (int splits_per_dim = ndim >= 8 ? 5 : 8; splits_per_dim < 15;
         splits_per_dim++) {

      Sub_regions<ndim> sub_regions(splits_per_dim);
      size_t num_regions = sub_regions.size;

      if (num_regions >= 35e6 || num_regions * 64 / INT_MAX >= 1.)
        break;

      Region_characteristics<ndim> characteristics(sub_regions.size);
      Region_estimates<ndim> estimates(sub_regions.size);

      Cubature_rules<ndim> rules;
      rules.set_device_volume(vol.lows, vol.highs);

      int iteration = 0;
      bool compute_relerr_error_reduction = false;
      cuhreResult<double> iter =
        rules.template apply_cubature_integration_rules<F>(
          d_integrand,
          &sub_regions,
          &estimates,
          &characteristics,
          compute_relerr_error_reduction);
      double estimate =
        custom_reduce<double>(estimates.integral_estimates, num_regions);
      // double errorest = reduction<double>(estimates.error_estimates,
      // num_regions);

      std::cout << "estimates:" << std::scientific << std::setprecision(15)
                << std::scientific << estimate << "," << num_regions
                << std::endl;
    }
  }

  auto q_ct1 = sycl::queue(sycl::gpu_selector());
  sycl::free(d_integrand, q_ct1);
}

template <typename F, int ndim>
void
call_cubature_rules(F integrand,
                    quad::Volume<double, ndim>& vol,
                    int num_repeats = 11)
{

  for (int i = 0; i < num_repeats; ++i) {
    for (int splits_per_dim = ndim >= 8 ? 5 : 8; splits_per_dim < 15;
         splits_per_dim++) {
      F* d_integrand = quad::make_gpu_integrand<F>(integrand);
      Sub_regions<ndim> sub_regions(splits_per_dim);
      size_t num_regions = sub_regions.size;

      if (num_regions >= 35e6 || num_regions * 64 / INT_MAX >= 1.)
        break;

      Region_characteristics<ndim> characteristics(sub_regions.size);
      Region_estimates<ndim> estimates(sub_regions.size);

      Cubature_rules<ndim> rules;
      rules.set_device_volume(vol.lows, vol.highs);

      bool compute_relerr_error_reduction = false;
      cuhreResult<double> iter =
        rules.template apply_cubature_integration_rules<F>(
          d_integrand,
          &sub_regions,
          &estimates,
          &characteristics,
          compute_relerr_error_reduction);

      double estimate =
        custom_reduce<double>(estimates.integral_estimates, num_regions);
      // double errorest = reduction<double>(estimates.error_estimates,
      // num_regions);

      std::cout << "estimates:" << std::scientific << std::setprecision(15)
                << std::scientific << estimate << "," << num_regions
                << std::endl;
      auto q_ct1 = sycl::queue(sycl::gpu_selector());
      sycl::free(d_integrand, q_ct1);
    }
  }
}

/*
    we are not keeping track of nFinished regions
    id, ndim, true_val, epsrel, epsabs, estimate, errorest, nregions,
   nFinishedRegions, status, time
*/

template <typename F, int ndim, bool use_custom = false, int debug = 0>
bool
clean_time_and_call(std::string id,
                    F integrand,
                    double epsrel,
                    double true_value,
                    char const* algname,
                    std::ostream& outfile,
                    bool relerr_classification = true)
{

  auto print_custom = [=](bool use_custom_flag) {
    std::string to_print = use_custom_flag == true ? "custom" : "library";
    return to_print;
  };

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;

  double constexpr epsabs = 1.0e-40;
  bool good = false;
  Workspace<ndim, use_custom> workspace;
  quad::Volume<double, ndim> vol;

  for (int i = 0; i < 2; i++) {
    auto const t0 = std::chrono::high_resolution_clock::now();
    size_t partitions_per_axis = 2;
    if (ndim < 5)
      partitions_per_axis = 4;
    else if (ndim <= 10)
      partitions_per_axis = 2;
    else
      partitions_per_axis = 1;
    Sub_regions<ndim> sub_regions(partitions_per_axis);

    constexpr bool collect_iters = false;
    constexpr bool collect_sub_regions = false;
    constexpr bool predict_split = false;
    cuhreResult<double> result =
      workspace.template integrate<F,
                                   predict_split,
                                   collect_iters,
                                   collect_sub_regions,
                                   debug>(
        integrand, sub_regions, epsrel, epsabs, vol, relerr_classification);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

    if (result.status == 0) {
      good = true;
    }

    outfile.precision(17);
    if (i != 0.)
      outfile << std::fixed << std::scientific << "pagani"
              << "," << id << "," << ndim << "," << print_custom(use_custom)
              << "," << true_value << "," << epsrel << "," << epsabs << ","
              << result.estimate << "," << result.errorest << ","
              << result.nregions << "," << result.nFinishedRegions << ","
              << result.status << "," << dt.count() << std::endl;
  }
  return good;
}

void
print_header()
{
  std::cout << "id, ndim, integral, epsrel, epsabs, estimate, errorest, "
               "nregions, status, time\n";
}

template <typename F, int ndim>
double
execute_integrand(std::array<double, ndim> point, size_t num_invocations)
{
  const size_t num_blocks = 1024;
  const size_t num_threads = 64;

  F integrand;
  F* d_integrand = quad::cuda_copy_to_managed(integrand);

  double* d_point = quad::cuda_malloc<double>(point.size());
  quad::cuda_memcpy_to_device(d_point, point.data(), point.size());

  double* output = quad::cuda_malloc<double>(num_threads * num_blocks);
  auto q = sycl::queue(
    sycl::gpu_selector() /*, sycl::property::queue::enable_profiling{}*/);

  for (int i = 0; i < 10; ++i) {
    /*sycl::event e = */ q
      .submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range(sycl::range(num_blocks * num_threads),
                                        sycl::range(num_threads)),
                         [=](sycl::nd_item<1> item_ct1)
                           [[intel::reqd_sub_group_size(32)]] {
                             size_t tid = item_ct1.get_local_id(0) +
                                          item_ct1.get_local_range().get(0) *
                                            item_ct1.get_group(0);
                             gpu::cudaArray<double, ndim> point;

                             double start_val = .01;
#pragma unroll ndim
                             for (int i = 0; i < ndim; ++i) {
                               point[i] = start_val * (i + 1);
                             }

                             double total = 0.;
#pragma unroll 1
                             for (int i = 0; i < num_invocations; ++i) {
                               double res = gpu::apply(*d_integrand, point);
                               total += res;
                             }
                             output[tid] = total;
                           });
      })
      .wait();

    q.wait();

    // double time = (e.template
    // get_profiling_info<sycl::info::event_profiling::command_end>()  -
    //	     e.template
    //get_profiling_info<sycl::info::event_profiling::command_start>());
    // std::cout<<"time:"<<time/1.e6 << std::endl;
  }
  // std::cout<<"time---------\n";
  std::vector<double> host_output;
  host_output.resize(num_threads * num_blocks);
  // std::cout<<"vector size:"<<host_output.size()<<std::endl;
  quad::cuda_memcpy_to_host<double>(
    host_output.data(), output, host_output.size());

  double sum = 0.;
  for (int i = 0; i < num_threads * num_blocks; ++i)
    sum += host_output[i];

  sycl::free(output, q);
  sycl::free(d_integrand, q);
  sycl::free(d_point, q);
  return sum;
}

template <typename F, int ndim>
double
execute_integrand_at_points(size_t num_invocations)
{
  const size_t num_blocks = 1024;
  const size_t num_threads = 64;

  F integrand;
  F* d_integrand = quad::cuda_copy_to_managed(integrand);

  double* points = quad::cuda_malloc<double>(num_invocations * ndim);

  std::vector<double> h_points;
  srand(1);

  for (int i = 0; i < num_invocations * ndim; ++i)
    h_points.push_back(rand());
  quad::cuda_memcpy_to_device(points, h_points.data(), h_points.size());

  double* output = quad::cuda_malloc<double>(num_threads * num_blocks);
  auto q = sycl::queue(
    sycl::gpu_selector() /*, sycl::property::queue::enable_profiling{}*/);

  for (int i = 0; i < 10; ++i) {
    /*sycl::event e = */ q
      .submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range(sycl::range(num_blocks * num_threads),
                                        sycl::range(num_threads)),
                         [=](sycl::nd_item<1> item_ct1)
                           [[intel::reqd_sub_group_size(32)]] {
                             size_t tid = item_ct1.get_local_id(0) +
                                          item_ct1.get_local_range().get(0) *
                                            item_ct1.get_group(0);
                             gpu::cudaArray<double, ndim> point;

                             double total = 0.;
#pragma unroll 1
                             for (int i = 0; i < num_invocations; ++i) {

#pragma unroll ndim
                               for (int dim = 0; dim < ndim; ++dim) {
                                 point[dim] = points[i * ndim + dim];
                               }

                               double res = gpu::apply(*d_integrand, point);
                               total += res;
                             }
                             output[tid] = total;
                           });
      })
      .wait();

    q.wait();

    // double time = (e.template
    // get_profiling_info<sycl::info::event_profiling::command_end>()  -
    //	     e.template
    //get_profiling_info<sycl::info::event_profiling::command_start>());
    // std::cout<<"time:"<<time/1.e6 << std::endl;
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

  sycl::free(output, q);
  sycl::free(d_integrand, q);
  sycl::free(points, q);
  return sum;
}

#endif
