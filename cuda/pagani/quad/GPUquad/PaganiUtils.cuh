#ifndef PAGANI_UTILS_CUH
#define PAGANI_UTILS_CUH

#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaApply.cuh"
#include "cuda/pagani/quad/util/cudaArray.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
//#include "cuda/pagani/quad/GPUquad/Kernel.cuh"
#include "cuda/pagani/quad/GPUquad/Phases.cuh"
#include "cuda/pagani/quad/GPUquad/Sample.cuh"
#include "cuda/pagani/quad/util/cuhreResult.cuh"
#include "cuda/pagani/quad/GPUquad/Rule.cuh"

#include "cuda/pagani/quad/GPUquad/hybrid.cuh"
#include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "cuda/pagani/quad/GPUquad/Region_estimates.cuh"
#include "cuda/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "cuda/pagani/quad/GPUquad/Sub_region_filter.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/GPUquad/Sub_region_splitter.cuh"
//#include "cuda/pagani/quad/GPUquad/heuristic_classifier.cuh"
#include "cuda/pagani/quad/util/custom_functions.cuh"
#include "cuda/pagani/quad/GPUquad/Func_Eval.cuh"
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cuda_profiler_api.h>

__constant__ size_t dFEvalPerRegion;

template <size_t ndim, bool use_custom = false>
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

  using Reg_estimates = Region_estimates<ndim>;
  using Sub_regs = Sub_regions<ndim>;
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
    quad::Rule<double> rule;
    const int key = 0;
    const int verbose = 0;
    rule.Init(ndim, fEvalPerRegion, key, verbose, &constMem);
    generators = cuda_malloc<double>(sizeof(double) * ndim * fEvalPerRegion);

    size_t block_size = 64;
    quad::ComputeGenerators<double, ndim>
      <<<1, block_size>>>(generators, fEvalPerRegion, constMem);

    integ_space_lows = cuda_malloc<double>(ndim);
    integ_space_highs = cuda_malloc<double>(ndim);

    set_device_volume();
  }

  void
  set_device_volume(double const* lows = nullptr, double const* highs = nullptr)
  {

    if (lows == nullptr && highs == nullptr) {
      std::array<double, ndim> _lows = {0.};
      std::array<double, ndim> _highs;
      std::fill_n(_highs.begin(), ndim, 1.);

      cuda_memcpy_to_device<double>(integ_space_highs, _highs.data(), ndim);
      cuda_memcpy_to_device<double>(integ_space_lows, _lows.data(), ndim);
    } else {
      cuda_memcpy_to_device<double>(integ_space_highs, highs, ndim);
      cuda_memcpy_to_device<double>(integ_space_lows, lows, ndim);
    }
  }

  ~Cubature_rules()
  {
    cudaFree(generators);
    cudaFree(integ_space_lows);
    cudaFree(integ_space_highs);
  }

  void
  Print_func_evals(quad::Func_Evals<ndim> fevals,
                   double* ests,
                   double* errs,
                   const size_t num_regions)
  {
    if (num_regions >= 1024)
      return;

    constexpr size_t num_fevals = pagani::CuhreFuncEvalsPerRegion<ndim>();

    auto print_reg = [=](const Bounds* sub_region) {
      for (size_t dim = 0; dim < ndim; ++dim) {
        rfevals.outfile << std::scientific << sub_region[dim].lower << ","
                        << sub_region[dim].upper << ",";
      }
    };

    auto print_global_bounds = [=](const GlobalBounds* sub_region) {
      for (size_t dim = 0; dim < ndim; ++dim) {
        rfevals.outfile << std::scientific << sub_region[dim].unScaledLower
                        << "," << sub_region[dim].unScaledUpper << ",";
      }
    };

    auto print_feval_point = [=](double* x) {
      for (size_t dim = 0; dim < ndim; ++dim) {
        rfevals.outfile << std::scientific << x[dim] << ",";
      }
    };

    for (size_t reg = 0; reg < num_regions; ++reg) {
      for (int feval = 0; feval < fevals.num_fevals; ++feval) {
        size_t index = reg * fevals.num_fevals + feval;

        rfevals.outfile << reg << "," << fevals[index].feval_index << ",";

        print_reg(fevals[index].region_bounds);
        print_global_bounds(fevals[index].global_bounds);
        print_feval_point(fevals[index].point);

        rfevals.outfile << std::scientific << fevals[index].feval << ",";
        rfevals.outfile << std::scientific << ests[reg] << "," << errs[reg];
        rfevals.outfile << std::endl;
      }
    }
  }

  void
  Print_region_evals(double* ests, double* errs, const size_t num_regions)
  {
    for (size_t reg = 0; reg < num_regions; ++reg) {
      rregions.outfile << reg << ",";
      rregions.outfile << std::scientific << ests[reg] << "," << errs[reg];
      rregions.outfile << std::endl;
    }
  }

  void
  print_generators(double* d_generators)
  {
    rgenerators.outfile << "i, gen" << std::endl;
    double* h_generators =
      new double[ndim * pagani::CuhreFuncEvalsPerRegion<ndim>()];
    cuda_memcpy_to_host<double>(h_generators,
                                d_generators,
                                ndim * pagani::CuhreFuncEvalsPerRegion<ndim>());

    for (int i = 0; i < ndim * pagani::CuhreFuncEvalsPerRegion<ndim>(); ++i) {
      rgenerators.outfile << i << "," << std::scientific << h_generators[i]
                          << std::endl;
    }
    delete[] h_generators;
  }

  template <int debug = 0>
  void
  print_verbose(double* d_generators,
                quad::Func_Evals<ndim>& dfevals,
                const Reg_estimates& estimates)
  {

    if constexpr (debug >= 1) {
      print_generators(d_generators);

      constexpr size_t num_fevals = pagani::CuhreFuncEvalsPerRegion<ndim>();
      const size_t num_regions = estimates.size;

      double* ests = new double[num_regions];
      double* errs = new double[num_regions];

      cuda_memcpy_to_host<double>(
        ests, estimates.integral_estimates, num_regions);
      cuda_memcpy_to_host<double>(errs, estimates.error_estimates, num_regions);

      Print_region_evals(ests, errs, num_regions);

      if constexpr (debug >= 2) {
        quad::Func_Evals<ndim>* hfevals = new quad::Func_Evals<ndim>;
        hfevals->fevals_list = new quad::Feval<ndim>[num_regions * num_fevals];
        cuda_memcpy_to_host<quad::Feval<ndim>>(
          hfevals->fevals_list, dfevals.fevals_list, num_regions * num_fevals);
        Print_func_evals(*hfevals, ests, errs, num_regions);
        delete[] hfevals->fevals_list;
        delete hfevals;
        cudaFree(dfevals.fevals_list);
      }

      delete[] ests;
      delete[] errs;
    }
  }

  template <int dim>
  void
  Setup_cubature_integration_rules()
  {
    size_t fEvalPerRegion = pagani::CuhreFuncEvalsPerRegion<dim>();
    quad::Rule<double> rule;
    const int key = 0;
    const int verbose = 0;
    rule.Init(dim, fEvalPerRegion, key, verbose, &constMem);
    generators = cuda_malloc<double>(sizeof(double) * dim * fEvalPerRegion);

    size_t block_size = 64;
    quad::ComputeGenerators<double, dim>
      <<<1, block_size>>>(generators, fEvalPerRegion, constMem);
  }

  template <typename IntegT>
  cuhreResult<double>
  apply_cubature_integration_rules(const IntegT& integrand,
                                   const Sub_regs& subregions,
                                   bool compute_error = true)
  {

    IntegT* d_integrand = make_gpu_integrand<IntegT>(integrand);

    size_t num_regions = subregions.size;
    Region_characteristics<ndim> region_characteristics(num_regions);
    Region_estimates<ndim> subregion_estimates(num_regions);

    set_device_array<int>(
      region_characteristics.active_regions, num_regions, 1);

    size_t num_blocks = num_regions;
    constexpr size_t block_size = 64;

    double epsrel = 1.e-3, epsabs = 1.e-12;

    quad::INTEGRATE_GPU_PHASE1<IntegT, double, ndim, block_size>
      <<<num_blocks, block_size>>>(d_integrand,
                                   subregions.dLeftCoord,
                                   subregions.dLength,
                                   num_regions,
                                   subregion_estimates.integral_estimates,
                                   subregion_estimates.error_estimates,
                                   region_characteristics.active_regions,
                                   region_characteristics.sub_dividing_dim,
                                   epsrel,
                                   epsabs,
                                   constMem,
                                   integ_space_lows,
                                   integ_space_highs,
                                   0,
                                   generators);
    cudaDeviceSynchronize();

    cuhreResult<double> res;
    res.estimate = reduction<double, use_custom>(
      subregion_estimates.integral_estimates, num_regions);
    res.errorest = compute_error ?
                     reduction<double, use_custom>(
                       subregion_estimates.error_estimates, num_regions) :
                     std::numeric_limits<double>::infinity();

    return res;
  }

  template <typename IntegT, int debug = 0>
  cuhreResult<double>
  apply_cubature_integration_rules(
    IntegT* d_integrand,
    int it,
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
        cuda_malloc<quad::Feval<ndim>>(num_regions * num_fevals);
    }

    set_device_array<int>(
      region_characteristics.active_regions, num_regions, 1.);

    size_t num_blocks = num_regions;
    constexpr size_t block_size = 64;

    double epsrel = 1.e-3, epsabs = 1.e-12;
    cudaProfilerStart();
    quad::INTEGRATE_GPU_PHASE1<IntegT, double, ndim, block_size, debug>
      <<<num_blocks, block_size>>>(d_integrand,
                                   subregions.dLeftCoord,
                                   subregions.dLength,
                                   num_regions,
                                   subregion_estimates.integral_estimates,
                                   subregion_estimates.error_estimates,
                                   region_characteristics.active_regions,
                                   region_characteristics.sub_dividing_dim,
                                   epsrel,
                                   epsabs,
                                   constMem,
                                   integ_space_lows,
                                   integ_space_highs,
                                   generators,
                                   dfevals);
    cudaDeviceSynchronize();
    cudaProfilerStop();

    print_verbose<debug>(generators, dfevals, subregion_estimates);
    /*if constexpr(debug >= 2){

            constexpr size_t num_fevals =
    pagani::CuhreFuncEvalsPerRegion<ndim>(); hfevals = new
    quad::Func_Evals<ndim>; hfevals->fevals_list = new
    quad::Feval<ndim>[num_regions*num_fevals]; double* ests = new
    double[num_regions]; double* errs = new double[num_regions];

            cuda_memcpy_to_host<double>(ests,
    subregion_estimates.integral_estimates, num_regions);
            cuda_memcpy_to_host<double>(errs,
    subregion_estimates.error_estimates, num_regions);
            cuda_memcpy_to_host<quad::Feval<ndim>>(hfevals->fevals_list,
    dfevals.fevals_list, num_regions*num_fevals);

            Print_region_evals(ests, errs, num_regions);

            Print_func_evals(*hfevals, ests, errs, num_regions);
            delete[] hfevals->fevals_list;
            delete hfevals;
            delete[] ests;
            delete[] errs;
            cudaFree(dfevals.fevals_list);

            auto print_generators = [=](double* d_generators){
                    rgenerators.outfile << "i, gen" << std::endl;
                    double* h_generators = new double[ndim *
    pagani::CuhreFuncEvalsPerRegion<ndim>()];
                    cuda_memcpy_to_host<double>(h_generators, d_generators, ndim
    * pagani::CuhreFuncEvalsPerRegion<ndim>()); for(int i=0; i < ndim *
    pagani::CuhreFuncEvalsPerRegion<ndim>(); ++i){ rgenerators.outfile << i <<
    "," << std::scientific << h_generators[i] << std::endl;
                    }
                    delete[] h_generators;
            };
            print_generators(generators);
    }*/

    cuhreResult<double> res;
    res.estimate = reduction<double, use_custom>(
      subregion_estimates.integral_estimates, num_regions);
    res.errorest = compute_error ?
                     reduction<double, use_custom>(
                       subregion_estimates.error_estimates, num_regions) :
                     std::numeric_limits<double>::infinity();
    return res;
  }

  Structures<double> constMem;
  double* generators = nullptr;

  double* integ_space_lows = nullptr;
  double* integ_space_highs = nullptr;
};

template <size_t ndim, bool use_custom = false>
cuhreResult<double>
compute_finished_estimates(const Region_estimates<ndim>& estimates,
                           const Region_characteristics<ndim>& classifiers,
                           const cuhreResult<double>& iter)
{
  cuhreResult<double> finished;
  finished.estimate =
    iter.estimate -
    dot_product<int, double, use_custom>(
      classifiers.active_regions, estimates.integral_estimates, estimates.size);
  ;
  finished.errorest =
    iter.errorest -
    dot_product<int, double, use_custom>(
      classifiers.active_regions, estimates.error_estimates, estimates.size);
  ;
  return finished;
}

bool
accuracy_reached(double epsrel, double epsabs, double estimate, double errorest)
{
  if (errorest / estimate <= epsrel || errorest <= epsabs)
    return true;
  return false;
}

bool
accuracy_reached(double epsrel, double epsabs, cuhreResult<double> res)
{
  if (res.errorest / res.estimate <= epsrel || res.errorest <= epsabs)
    return true;
  return false;
}

template <typename IntegT, int ndim>
cuhreResult<double>
pagani_clone(const IntegT& integrand,
             Sub_regions<ndim>& subregions,
             double epsrel = 1.e-3,
             double epsabs = 1.e-12,
             bool relerr_classification = true)
{
  using Reg_estimates = Region_estimates<ndim>;
  using Sub_regs = Sub_regions<ndim>;
  using Regs_characteristics = Region_characteristics<ndim>;
  using Res = cuhreResult<double>;
  using Filter = Sub_regions_filter<ndim>;
  using Splitter = Sub_region_splitter<ndim>;
  Reg_estimates prev_iter_estimates;

  Res cummulative;
  Cubature_rules<ndim> cubature_rules;
  Heuristic_classifier<ndim> hs_classify(epsrel, epsabs);
  bool accuracy_termination = false;
  IntegT* d_integrand = make_gpu_integrand<IntegT>(integrand);

  for (size_t it = 0; it < 700 && !accuracy_termination; it++) {
    size_t num_regions = subregions.size;
    Regs_characteristics classifiers(num_regions);
    Reg_estimates estimates(num_regions);

    Res iter = cubature_rules.apply_cubature_integration_rules(
      d_integrand, subregions, estimates, classifiers);
    computute_two_level_errorest<ndim>(
      estimates, prev_iter_estimates, classifiers, relerr_classification);
    // iter_res.estimate = reduction<double>(estimates.integral_estimates,
    // num_regions);
    iter.errorest = reduction<double>(estimates.error_estimates, num_regions);

    accuracy_termination =
      accuracy_reached(epsrel,
                       epsabs,
                       std::abs(cummulative.estimate + iter.estimate),
                       cummulative.errorest + iter.errorest);

    // where are the conditions to hs_classify (gpu mem and convergence?)
    if (!accuracy_termination) {

      // 1. store the latest estimate so that we can check whether estimate
      // convergence happens
      hs_classify.store_estimate(cummulative.estimate + iter.estimate);

      // 2.get the actual finished estimates, needs to happen before hs
      // heuristic classification
      Res finished;
      finished.estimate =
        iter.estimate - dot_product<int, double>(classifiers.active_regions,
                                                 estimates.integral_estimates,
                                                 num_regions);
      finished.errorest =
        iter.errorest - dot_product<int, double>(classifiers.active_regions,
                                                 estimates.error_estimates,
                                                 num_regions);

      // 3. try classification
      // THIS SEEMS WRONG WHY WE PASS ITER.ERROREST TWICE? LAST PARAM SHOULD BE
      // TOTAL FINISHED ERROREST, SO CUMMULATIVE.ERROREST
      Classification_res hs_results =
        hs_classify.classify(classifiers.active_regions,
                             estimates.error_estimates,
                             num_regions,
                             iter.errorest,
                             finished.errorest,
                             iter.errorest);

      // 4. check if classification actually happened or was successful
      bool hs_classify_success =
        hs_results.pass_mem && hs_results.pass_errorest_budget;
      // printf("hs_results will leave %lu regions active\n",
      // hs_results.num_active);

      if (hs_classify_success) {
        // 5. if classification happened and was successful, update finished
        // estimates
        classifiers.active_regions = hs_results.active_flags;
        finished.estimate =
          iter.estimate - dot_product<int, double>(classifiers.active_regions,
                                                   estimates.integral_estimates,
                                                   num_regions);

        finished.errorest = hs_results.finished_errorest;
      }

      // 6. update cummulative estimates with finished contributions
      cummulative.estimate += finished.estimate;
      cummulative.errorest += finished.errorest;

      // printf("num regions pre filtering:%lu\n", subregions->size);
      // 7. Filter out finished regions
      Filter region_errorest_filter(num_regions);
      num_regions = region_errorest_filter.filter(
        subregions, classifiers, estimates, prev_iter_estimates);
      // printf("num regions after filtering:%lu\n", subregions->size);
      quad::CudaCheckError();
      // split regions
      // printf("num regions pre split:%lu\n", subregions->size);
      Splitter reg_splitter(num_regions);
      reg_splitter.split(subregions, classifiers);
      // printf("num regions after split:%lu\n", subregions->size);
    } else {
      cummulative.estimate += iter.estimate;
      cummulative.errorest += iter.errorest;
    }
  }
  return cummulative;
}

#endif
