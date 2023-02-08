#ifndef KOKKOS_PAGANI_UTILS_CUH
#define KOKKOS_PAGANI_UTILS_CUH

#include "common/kokkos/Volume.cuh"
#include "common/kokkos/cudaApply.cuh"
#include "common/kokkos/cudaMemoryUtil.h"
#include "common/integration_result.hh"
#include "common/kokkos/thrust_utils.cuh"

#include "kokkos/pagani/quad/GPUquad/Phases.cuh"
#include "kokkos/pagani/quad/GPUquad/Rule.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_regions.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_region_filter.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_region_splitter.cuh"
#include "kokkos/pagani/quad/GPUquad/Func_Eval.cuh"
#include "kokkos/pagani/quad/quad.h"

#include <stdlib.h>
#include <fstream>
#include <string>

template <typename T, size_t ndim, bool use_custom = false>
class Cubature_rules {
public:
  // integrator requires constMem structure and generators array (those two can
  // and should be combined into one) ideally we dont have quad::Rule all, we
  // should only use it to intantiate constMem and be done with it requires
  // restructuring of Rule.cuh

  // PaganiWrapper contructor essentially sets up the structures to integrate
  // some regions the target regions to integrate can then be passes as a
  // parameter, with type SubRegions<int ndim>

  // actual integration requires more though

  using Reg_estimates = Region_estimates<T, ndim>;
  using Sub_regs = Sub_regions<T, ndim>;
  using Regs_characteristics = Region_characteristics<ndim>;
  Recorder<true> rfevals;
  Recorder<true> rregions;
  Recorder<true> rgenerators;

  Cubature_rules()
  {
    rfevals.outfile.open("cuda_fevals.csv");
    rgenerators.outfile.open("cuda_generators.csv");
    rregions.outfile.open("cuda_regions.csv");

    auto print_header = [=]() {
      rfevals.outfile << "reg, fid,";
      for (size_t dim = 0; dim < ndim; ++dim)
        rfevals.outfile << "dim" + std::to_string(dim) << +"low"
                        << ","
                        << "dim" + std::to_string(dim) + "high,";

      for (size_t dim = 0; dim < ndim; ++dim)
        rfevals.outfile << "gdim" + std::to_string(dim) << +"low"
                        << ","
                        << "gdim" + std::to_string(dim) + "high"
                        << ",";

      for (size_t dim = 0; dim < ndim; ++dim)
        rfevals.outfile << "dim" + std::to_string(dim) << ",";

      rfevals.outfile << std::scientific << "feval, estimate, errorest"
                      << std::endl;
    };

    print_header();
    constexpr size_t fEvalPerRegion = pagani::CuhreFuncEvalsPerRegion<ndim>();
    quad::Rule<T> rule;
    const int key = 0;
    const int verbose = 0;
    rule.Init(ndim, fEvalPerRegion, key, verbose, &constMem);
    generators = quad::cuda_malloc<T>(ndim * fEvalPerRegion);

    quad::ComputeGenerators<T, ndim>(generators, fEvalPerRegion, constMem);

    integ_space_lows = quad::cuda_malloc<T>(ndim);
    integ_space_highs = quad::cuda_malloc<T>(ndim);

    set_device_volume();
  }

  void
  set_device_volume(T const* lows = nullptr, T const* highs = nullptr)
  {

    auto _lows = Kokkos::create_mirror_view(integ_space_lows);
    auto _highs = Kokkos::create_mirror_view(integ_space_highs);

    if (lows == nullptr && highs == nullptr) {
      std::fill_n(_highs.data(), ndim, 1.);
      Kokkos::deep_copy(integ_space_highs, _highs);
      Kokkos::deep_copy(integ_space_lows, _lows);

    } else {

      for (int dim = 0; dim < ndim; ++dim) {
        _lows[dim] = lows[dim];
        _highs[dim] = highs[dim];
      }

      Kokkos::deep_copy(integ_space_highs, _highs);
      Kokkos::deep_copy(integ_space_lows, _lows);
    }
  }

  ~Cubature_rules() {}

  void
  Print_region_evals(T* ests, T* errs, const size_t num_regions)
  {
    for (size_t reg = 0; reg < num_regions; ++reg) {
      rregions.outfile << reg << ",";
      rregions.outfile << std::scientific << ests[reg] << "," << errs[reg];
      rregions.outfile << std::endl;
    }
  }

  void
  print_generators(ViewVectorDouble d_generators)
  {
    rgenerators.outfile << "i, gen" << std::endl;
    auto h_generators = Kokkos::create_mirror_view(d_generators);
    Kokkos::deep_copy(h_generators, d_generators);

    for (int i = 0; i < ndim * pagani::CuhreFuncEvalsPerRegion<ndim>(); ++i) {
      rgenerators.outfile << i << "," << std::scientific << h_generators[i]
                          << std::endl;
    }
  }

  template <int debug = 0>
  void
  print_verbose(T* d_generators,
                quad::Func_Evals<ndim>& dfevals,
                const Reg_estimates& estimates)
  {

    if constexpr (debug >= 1) {
      print_generators(d_generators);

      constexpr size_t num_fevals = pagani::CuhreFuncEvalsPerRegion<ndim>();
      const size_t num_regions = estimates.size;

      auto ests = Kokkos::create_mirror_view(estimates.integral_estimates);
      auto errs = Kokkos::create_mirror_view(estimates.error_estimates);

      Kokkos::deep_copy(ests, estimates.integral_estimates);
      Kokkos::deep_copy(errs, estimates.error_estimates);
      Print_region_evals(ests, errs, num_regions);

      if constexpr (debug >= 2) {
        auto hfevals = Kokkos::create_mirror_view(dfevals.fevals_list);
        Print_func_evals(hfevals, ests, errs, num_regions);
      }
    }
  }

  /*template <typename IntegT>
  numint::integration_result
  apply_cubature_integration_rules(const IntegT& integrand,
                                   const Sub_regs& subregions,
                                   bool compute_error = true)
  {

    IntegT* d_integrand = quad::make_gpu_integrand<IntegT>(integrand);

    size_t num_regions = subregions.size;
    Region_characteristics<ndim> region_characteristics(num_regions);
    Region_estimates<T, ndim> subregion_estimates(num_regions);

    quad::set_device_array<int>(
      region_characteristics.active_regions, num_regions, 1);

    size_t num_blocks = num_regions;
    constexpr size_t block_size = 64;

    T epsrel = 1.e-3, epsabs = 1.e-12;

    quad::INTEGRATE_GPU_PHASE1<IntegT, T, ndim, block_size>
      (d_integrand,
                                   subregions.dLeftCoord.data(),
                                   subregions.dLength.data(),
                                   num_regions,
                                   subregion_estimates.integral_estimates.data(),
                                   subregion_estimates.error_estimates.data(),
                                   // region_characteristics.active_regions,
                                   region_characteristics.sub_dividing_dim.data(),
                                   epsrel,
                                   epsabs,
                                   constMem,
                                   integ_space_lows.data(),
                                   integ_space_highs.data(),
                                   0,
                                   generators.data());

    numint::integration_result res;
    res.estimate = reduction<T, use_custom>(
      subregion_estimates.integral_estimates, num_regions);
    res.errorest = compute_error ?
                     reduction<T, use_custom>(
                       subregion_estimates.error_estimates, num_regions) :
                     std::numeric_limits<T>::infinity();

    return res;
  }
*/
  template <typename IntegT, int debug = 0>
  numint::integration_result
  apply_cubature_integration_rules(
    IntegT* d_integrand,
    const Sub_regs& subregions,
    const Reg_estimates& subregion_estimates,
    const Regs_characteristics& region_characteristics,
    bool compute_error = false)
  {
    size_t num_regions = subregions.size;
    quad::Func_Evals<ndim> dfevals;

    if constexpr (debug >= 2) {
      constexpr size_t num_fevals = pagani::CuhreFuncEvalsPerRegion<ndim>();
      dfevals.fevals_list =
        quad::cuda_malloc<quad::Feval<ndim>>(num_regions * num_fevals);
    }

    quad::set_device_array<int>(
      region_characteristics.active_regions.data(), num_regions, 1.);

    constexpr size_t block_size = 64;
    T epsrel = 1.e-3, epsabs = 1.e-12;

    quad::INTEGRATE_GPU_PHASE1<IntegT, T, ndim, block_size, debug>(
      d_integrand,
      subregions.dLeftCoord.data(),
      subregions.dLength.data(),
      num_regions,
      subregion_estimates.integral_estimates.data(),
      subregion_estimates.error_estimates.data(),
      region_characteristics.sub_dividing_dim.data(),
      constMem,
      integ_space_lows.data(),
      integ_space_highs.data(),
      generators.data(),
      dfevals);

    print_verbose<debug>(generators.data(), dfevals, subregion_estimates);
    std::cout << "about to do reduction" << std::endl;
    numint::integration_result res;
    res.estimate = reduction<T, use_custom>(
      subregion_estimates.integral_estimates, num_regions);
    res.errorest = compute_error ?
                     reduction<T, use_custom>(
                       subregion_estimates.error_estimates, num_regions) :
                     std::numeric_limits<T>::infinity();
    return res;
  }

  template <int dim>
  void
  Setup_cubature_integration_rules()
  {
    size_t fEvalPerRegion = pagani::CuhreFuncEvalsPerRegion<dim>();
    quad::Rule<T> rule;
    const int key = 0;
    const int verbose = 0;
    rule.Init(dim, fEvalPerRegion, key, verbose, &constMem);
    generators = quad::cuda_malloc<T>(sizeof(T) * dim * fEvalPerRegion);

    size_t block_size = 64;
    quad::ComputeGenerators<T, dim>
      <<<1, block_size>>>(generators, fEvalPerRegion, constMem);
  }

  Structures<T> constMem;
  Kokkos::View<T*, Kokkos::CudaSpace> generators;

  Kokkos::View<T*, Kokkos::CudaSpace> integ_space_lows;
  Kokkos::View<T*, Kokkos::CudaSpace> integ_space_highs;
};

template <typename T, size_t ndim, bool use_custom = false>
numint::integration_result
compute_finished_estimates(const Region_estimates<T, ndim>& estimates,
                           const Region_characteristics<ndim>& classifiers,
                           const numint::integration_result& iter)
{
  numint::integration_result finished;
  finished.estimate =
    iter.estimate - dot_product<int, T, use_custom>(
                      classifiers.active_regions, estimates.integral_estimates);
  finished.errorest =
    iter.errorest - dot_product<int, T, use_custom>(classifiers.active_regions,
                                                    estimates.error_estimates);
  return finished;
}

template <typename T>
bool
accuracy_reached(T epsrel, T epsabs, T estimate, T errorest)
{
  if (errorest / estimate <= epsrel || errorest <= epsabs)
    return true;
  return false;
}

template <typename T>
bool
accuracy_reached(T epsrel, T epsabs, numint::integration_result res)
{
  if (res.errorest / res.estimate <= epsrel || res.errorest <= epsabs)
    return true;
  return false;
}

#endif
