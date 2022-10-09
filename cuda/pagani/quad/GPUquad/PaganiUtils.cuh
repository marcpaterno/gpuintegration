#ifndef PAGANI_UTILS_CUH
#define PAGANI_UTILS_CUH

#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaApply.cuh"
#include "cuda/pagani/quad/util/cudaArray.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
// #include "cuda/pagani/quad/GPUquad/Kernel.cuh"
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
// #include "cuda/pagani/quad/GPUquad/heuristic_classifier.cuh"
#include "cuda/pagani/quad/util/custom_functions.cuh"
#include "cuda/pagani/quad/GPUquad/Func_Eval.cuh"
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cuda_profiler_api.h>

__constant__ size_t dFEvalPerRegion;

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

    auto print_aser = [=]() {
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
    generators = cuda_malloc<T>(sizeof(T) * ndim * fEvalPerRegion);

    size_t block_size = 64;
    quad::ComputeGenerators<T, ndim>
      <<<1, block_size>>>(generators, fEvalPerRegion, constMem);

    integ_space_lows = cuda_malloc<T>(ndim);
    integ_space_highs = cuda_malloc<T>(ndim);

    set_device_volume();
  }

  void
  set_device_volume(T const* lows = nullptr, T const* highs = nullptr)
  {

    if (lows == nullptr && highs == nullptr) {
      std::array<T, ndim> _lows = {0.};
      std::array<T, ndim> _highs;
      std::fill_n(_highs.begin(), ndim, 1.);

      cuda_memcpy_to_device<T>(integ_space_highs, _highs.data(), ndim);
      cuda_memcpy_to_device<T>(integ_space_lows, _lows.data(), ndim);
    } else {
      cuda_memcpy_to_device<T>(integ_space_highs, highs, ndim);
      cuda_memcpy_to_device<T>(integ_space_lows, lows, ndim);
    }
  }

  ~Cubature_rules()
  {
    cudaFree(generators);
    cudaFree(integ_space_lows);
    cudaFree(integ_space_highs);
    cudaFree(constMem.gpuG);
    cudaFree(constMem.cRuleWt);
    cudaFree(constMem.GPUScale);
    cudaFree(constMem.GPUNorm);
    cudaFree(constMem.gpuGenPos);
    cudaFree(constMem.gpuGenPermGIndex);
    cudaFree(constMem.gpuGenPermVarCount);
    cudaFree(constMem.gpuGenPermVarStart);
    cudaFree(constMem.cGeneratorCount);
  }

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
  print_generators(T* d_generators)
  {
    rgenerators.outfile << "i, gen" << std::endl;
    T* h_generators = new T[ndim * pagani::CuhreFuncEvalsPerRegion<ndim>()];
    cuda_memcpy_to_host<T>(h_generators,
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
  print_verbose(T* d_generators,
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


    
    template<typename IntegT>
    cuhreResult<T> apply_cubature_integration_rules(const IntegT& integrand, const  Sub_regs& subregions, bool compute_error = true){
        
        IntegT* d_integrand =  make_gpu_integrand<IntegT>(integrand);
        
        size_t num_regions = subregions.size;
        Region_characteristics<ndim> region_characteristics(num_regions);
        Region_estimates<T, ndim> subregion_estimates(num_regions);
        
        set_device_array<int>(region_characteristics.active_regions, num_regions, 1);

        size_t num_blocks = num_regions;
        constexpr size_t block_size = 64;
        
        T epsrel = 1.e-3, epsabs = 1.e-12;
        
        quad::INTEGRATE_GPU_PHASE1<IntegT, T, ndim, block_size><<<num_blocks, block_size>>>
            (d_integrand, 
            subregions.dLeftCoord, 
            subregions.dLength, num_regions, 
            subregion_estimates.integral_estimates,
            subregion_estimates.error_estimates, 
            //region_characteristics.active_regions, 
            region_characteristics.sub_dividing_dim, 
            epsrel, 
            epsabs, 
            constMem, 
            integ_space_lows, 
            integ_space_highs, 
            0, 
            generators);
        cudaDeviceSynchronize();
        
        cuhreResult<T> res;
        res.estimate = reduction<T, use_custom>(subregion_estimates.integral_estimates, num_regions);
        res.errorest = compute_error ? reduction<T, use_custom>(subregion_estimates.error_estimates, num_regions) : std::numeric_limits<T>::infinity();
        
        return res;
    }
    
    template<typename IntegT, int debug = 0>
    cuhreResult<T> 
    apply_cubature_integration_rules(IntegT* d_integrand,
		int it, 
        const Sub_regs& subregions, 
        const Reg_estimates& subregion_estimates, 
        const Regs_characteristics& region_characteristics, 
        bool compute_error = false)
    {
		size_t num_regions = subregions.size;
		quad::Func_Evals<ndim> dfevals;
		
        if constexpr(debug >= 2){
			constexpr size_t num_fevals = pagani::CuhreFuncEvalsPerRegion<ndim>();
			dfevals.fevals_list = cuda_malloc<quad::Feval<ndim>>(num_regions*num_fevals); 
        }
        
        set_device_array<int>(region_characteristics.active_regions, num_regions, 1.);
        
        size_t num_blocks = num_regions;
        constexpr size_t block_size = 64;
        
        T epsrel = 1.e-3, epsabs = 1.e-12;
		cudaProfilerStart();
        quad::INTEGRATE_GPU_PHASE1<IntegT, T, ndim, block_size, debug><<<num_blocks, block_size>>>
            (d_integrand, 
            subregions.dLeftCoord, 
            subregions.dLength, num_regions, 
            subregion_estimates.integral_estimates,
            subregion_estimates.error_estimates, 
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
		
        cuhreResult<T> res;
        res.estimate = reduction<T, use_custom>(subregion_estimates.integral_estimates, num_regions);
        res.errorest = compute_error ? reduction<T, use_custom>(subregion_estimates.error_estimates, num_regions) : std::numeric_limits<T>::infinity();        
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
    generators = cuda_malloc<T>(sizeof(T) * dim * fEvalPerRegion);

    size_t block_size = 64;
    quad::ComputeGenerators<T, dim>
      <<<1, block_size>>>(generators, fEvalPerRegion, constMem);
  }

  Structures<T> constMem;
  T* generators = nullptr;

  T* integ_space_lows = nullptr;
  T* integ_space_highs = nullptr;
};

template <typename T, size_t ndim, bool use_custom = false>
cuhreResult<T>
compute_finished_estimates(const Region_estimates<T, ndim>& estimates,
                           const Region_characteristics<ndim>& classifiers,
                           const cuhreResult<T>& iter)
{
  cuhreResult<T> finished;
  finished.estimate =
    iter.estimate -
    dot_product<int, T, use_custom>(
      classifiers.active_regions, estimates.integral_estimates, estimates.size);
  ;
  finished.errorest =
    iter.errorest - dot_product<int, T, use_custom>(classifiers.active_regions,
                                                    estimates.error_estimates,
                                                    estimates.size);
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
accuracy_reached(T epsrel, T epsabs, cuhreResult<T> res)
{
  if (res.errorest / res.estimate <= epsrel || res.errorest <= epsabs)
    return true;
  return false;
}

template <typename T, typename IntegT, int ndim>
cuhreResult<T>
pagani_clone(const IntegT& integrand,
             Sub_regions<T, ndim>& subregions,
             T epsrel = 1.e-3,
             T epsabs = 1.e-12,
             bool relerr_classification = true)
{
  using Reg_estimates = Region_estimates<T, ndim>;
  using Sub_regs = Sub_regions<T, ndim>;
  using Regs_characteristics = Region_characteristics<ndim>;
  using Res = cuhreResult<T>;
  using Filter = Sub_regions_filter<T, ndim>;
  using Splitter = Sub_region_splitter<T, ndim>;
  Reg_estimates prev_iter_estimates;

  Res cummulative;
  Cubature_rules<T, ndim> cubature_rules;
  Heuristic_classifier<T, ndim> hs_classify(epsrel, epsabs);
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
    iter.errorest = reduction<T>(estimates.error_estimates, num_regions);

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
        iter.estimate - dot_product<int, T>(classifiers.active_regions,
                                            estimates.integral_estimates,
                                            num_regions);
      finished.errorest =
        iter.errorest - dot_product<int, T>(classifiers.active_regions,
                                            estimates.error_estimates,
                                            num_regions);

      // 3. try classification
      // THIS SEEMS WRONG WHY WE PASS ITER.ERROREST TWICE? LAST PARAM SHOULD BE
      // TOTAL FINISHED ERROREST, SO CUMMULATIVE.ERROREST
      Classification_res<T> hs_results =
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
          iter.estimate - dot_product<int, T>(classifiers.active_regions,
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
