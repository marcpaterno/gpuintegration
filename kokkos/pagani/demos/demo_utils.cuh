#ifndef KOKKOS_PAGANI_DEMO_UTILS_CUH
#define KOKKOS_PAGANI_DEMO_UTILS_CUH

#include "common/integration_result.hh"
#include "kokkos/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "kokkos/pagani/quad/GPUquad/Workspace.cuh"
#include "common/kokkos/Volume.cuh"
#include "common/kokkos/cudaMemoryUtil.h"
#include "kokkos/pagani/quad/Cuhre.cuh"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

template <typename F, int ndim, bool use_custom = false>
void
call_cubature_rules(int num_repeats = 11)
{
  F integrand;
  quad::Volume<double, ndim> vol;
  F* d_integrand = quad::make_gpu_integrand<F>(integrand);
  
  for (int i = 0; i < num_repeats; ++i) {
    for (int splits_per_dim = ndim >= 8 ? 5 : 8; splits_per_dim < 15;
         splits_per_dim++) {

      Sub_regions<double, ndim> sub_regions(splits_per_dim);
      size_t num_regions = sub_regions.size;

      if (num_regions >= 35e6 || num_regions * 64 / INT_MAX >= 1.)
        break;

      Region_characteristics<ndim> characteristics(sub_regions.size);
      Region_estimates<double, ndim> estimates(sub_regions.size);

      Cubature_rules<double, ndim, use_custom> rules;
      rules.set_device_volume(vol.lows, vol.highs);

      bool compute_relerr_error_reduction = false;
      numint::integration_result iter =
        rules.template apply_cubature_integration_rules<F>(
          d_integrand,
          sub_regions,
          estimates,
          characteristics,
          compute_relerr_error_reduction);
		  
      double estimate =
        reduction<double, use_custom>(estimates.integral_estimates, num_regions);
       double errorest = reduction<double, use_custom>(estimates.error_estimates,
       num_regions);

      std::cout << "estimates:" << std::scientific << std::setprecision(15)
                << std::scientific << estimate << "," << num_regions
                << std::endl;
    }
  }

  Kokkos::kokkos_free(d_integrand);
}

template <typename F, int ndim, bool use_custom = false>
void
call_cubature_rules(F integrand,
                    quad::Volume<double, ndim>& vol,
                    int num_repeats = 11)
{
  
  for (int i = 0; i < num_repeats; ++i) {
    for (int splits_per_dim = ndim >= 8 ? 5 : 8; splits_per_dim < 15; splits_per_dim++) {
      F* d_integrand = quad::make_gpu_integrand<F>(integrand);
      Sub_regions<double, ndim> sub_regions(splits_per_dim);
      size_t num_regions = sub_regions.size;

      if (num_regions >= 35e6 || num_regions * 64 / INT_MAX >= 1.)
        break;

      Region_characteristics<ndim> characteristics(sub_regions.size);
      Region_estimates<double, ndim> estimates(sub_regions.size);

      Cubature_rules<double, ndim, use_custom> rules;
      rules.set_device_volume(vol.lows, vol.highs);
	
      bool compute_relerr_error_reduction = false;
      numint::integration_result iter =
        rules.template apply_cubature_integration_rules<F>(
          d_integrand,
          sub_regions,
          estimates,
          characteristics,
          compute_relerr_error_reduction);

      double estimate =
        reduction<double, use_custom>(estimates.integral_estimates, num_regions);
      double errorest = reduction<double, use_custom>(estimates.error_estimates, num_regions);

      std::cout << "estimates:" << std::scientific << std::setprecision(15)
                << std::scientific << iter.estimate << "," << num_regions
                << std::endl;
		Kokkos::kokkos_free(d_integrand);
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
time_and_call(std::string id,
                    F integrand,
                    double epsrel,
                    double true_value)
{
  
  auto print_custom = [=](bool use_custom_flag) {
    std::string to_print = use_custom_flag == true ? "custom" : "library";
    return to_print;
  };

  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;

  double constexpr epsabs = 1.0e-40;
  bool good = false;
  Workspace<double, ndim, use_custom> workspace;
  const quad::Volume<double, ndim> vol;

  for (int i = 0; i < 2; i++) {
    auto const t0 = std::chrono::high_resolution_clock::now();

    constexpr bool collect_iters = false;
    constexpr bool predict_split = false;
	bool relerr_classification = true;
    numint::integration_result result =
      workspace.template integrate<F,
                                   predict_split,
                                   collect_iters,
                                   debug>(
        integrand, epsrel, epsabs, vol, relerr_classification);
    MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;

    if (result.status == 0) {
      good = true;
    }

    std::cout.precision(17);
    if (i != 0.)
      std::cout << std::fixed << std::scientific << "pagani"
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

  ViewVectorDouble d_point("d_point", point.size());
  auto h_point = Kokkos::create_mirror_view(d_point);
  
  for(int i=0; i < point.size(); ++i)
	  h_point[i] = point[i];
  
  Kokkos::deep_copy(d_point, h_point);
  ViewVectorDouble output("d_point", num_threads * num_blocks);
 
  auto h_output = Kokkos::create_mirror_view(output);
  
  for (int i = 0; i < 10; ++i) {

    Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> mainKernelPolicy(num_blocks,
                                                                      num_threads);
    Kokkos::parallel_for(
      "Phase1",
      mainKernelPolicy,
      KOKKOS_LAMBDA(const member_type team_member) {
        size_t tid = team_member.team_rank() +
            team_member.team_size() *
            team_member.league_rank();
        gpu::cudaArray<double, ndim> point;

        double start_val = .01;
        for (int i = 0; i < ndim; ++i) {
            point[i] = start_val * (i + 1);
        }

        double total = 0.;
        for (size_t i = 0; i < num_invocations; ++i) {
            double res = gpu::apply(*d_integrand, point);
            total += res;
        }
         output[tid] = total;
	});
  }
  
  Kokkos::deep_copy(h_output, output);
  double sum = 0.;
  for (int i = 0; i < num_threads * num_blocks; ++i)
    sum += h_output[i];

  Kokkos::kokkos_free(d_integrand);
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

  ViewVectorDouble points = quad::cuda_malloc<double>(num_invocations * ndim);

  srand(1);
  auto h_points = Kokkos::create_mirror_view(points);
  for (size_t i = 0; i < num_invocations * ndim; ++i)
    h_points[i]=rand();
  Kokkos::deep_copy(points, h_points);
  ViewVectorDouble output("d_point", num_threads * num_blocks);
  auto h_output = Kokkos::create_mirror_view(output);

  Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> mainKernelPolicy(num_blocks,
                                                                      num_threads);
    Kokkos::parallel_for(
      "execute_integrand_at_points",
      mainKernelPolicy,
      KOKKOS_LAMBDA(const member_type team_member) {
                             size_t tid = team_member.team_rank() +
                                          team_member.team_size() *
                                            team_member.league_rank();
                             gpu::cudaArray<double, ndim> point;

                             double total = 0.;
                             for (size_t i = 0; i < num_invocations; ++i) {

                               for (int dim = 0; dim < ndim; ++dim) {
                                 point[dim] = points[i * ndim + dim];
                               }

                               double res = gpu::apply(*d_integrand, point);
                               total += res;
                             }
                             output[tid] = total;
     });

  
  Kokkos::deep_copy(h_output, output);

  double sum = 0.;
  for (int i = 0; i < num_threads * num_blocks; ++i)
    sum += h_output[i];

  Kokkos::kokkos_free(d_integrand);
  return sum;
}

#endif