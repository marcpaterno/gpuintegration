#ifndef PAGANI_UTILS_CUH
#define PAGANI_UTILS_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "dpct-exp/common/cuda/Volume.dp.hpp"
#include "dpct-exp/common/cuda/cudaApply.dp.hpp"
#include "dpct-exp/common/cuda/cudaArray.dp.hpp"
#include "dpct-exp/common/cuda/cudaUtil.h"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Phases.dp.hpp"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Sample.dp.hpp"
#include "dpct-exp/common/integration_result.hh"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Rule.dp.hpp"

#include "dpct-exp/cuda/pagani/quad/GPUquad/hybrid.dp.hpp"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Sub_regions.dp.hpp"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Region_estimates.dp.hpp"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Region_characteristics.dp.hpp"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Sub_region_filter.dp.hpp"
#include "dpct-exp/cuda/pagani/quad/quad.h"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Sub_region_splitter.dp.hpp"
#include "dpct-exp/common/cuda/custom_functions.dp.hpp"
#include "dpct-exp/cuda/pagani/quad/GPUquad/Func_Eval.dp.hpp"
#include <stdlib.h>
#include <fstream>
#include <string>
#include <chrono>

dpct::constant_memory<size_t, 0> dFEvalPerRegion;

void tempff(double* temp){
	size_t size = 64;
    auto q_ct1 =  sycl::queue(sycl::gpu_selector());

	q_ct1.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, 64) *
                         sycl::range(1, 1, 64),
                       sycl::range(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
			
			temp[0]=1.;
        });
    });
    q_ct1.wait_and_throw();
	std::cout<<"end of tempff"<<std::endl;
}

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
    generators = quad::cuda_malloc<T>(sizeof(T) * ndim * fEvalPerRegion);

    size_t block_size = 64;
    /*
    DPCT1049:152: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler& cgh) {
      auto generators_ct0 = generators;
      auto constMem_ct2 = constMem;

      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, block_size),
                                      sycl::range(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         quad::ComputeGenerators<T, ndim>(generators_ct0,
                                                    fEvalPerRegion,
                                                    constMem_ct2,
                                                    item_ct1);
                       });
    });

    integ_space_lows = quad::cuda_malloc<T>(ndim);
    integ_space_highs = quad::cuda_malloc<T>(ndim);

    set_device_volume();
  }

  void
  set_device_volume(T const* lows = nullptr, T const* highs = nullptr)
  {

    if (lows == nullptr && highs == nullptr) {
      std::array<T, ndim> _lows = {0.};
      std::array<T, ndim> _highs;
      std::fill_n(_highs.begin(), ndim, 1.);

      quad::cuda_memcpy_to_device<T>(integ_space_highs, _highs.data(), ndim);
      quad::cuda_memcpy_to_device<T>(integ_space_lows, _lows.data(), ndim);
    } else {
      quad::cuda_memcpy_to_device<T>(integ_space_highs, highs, ndim);
      quad::cuda_memcpy_to_device<T>(integ_space_lows, lows, ndim);
    }
  }

  ~Cubature_rules()
  {
	dpct::device_ext& dev_ct1 = dpct::get_current_device();
	sycl::queue& q_ct1 = dev_ct1.default_queue();
    sycl::free(generators, q_ct1);
    sycl::free(integ_space_lows, q_ct1);
    sycl::free(integ_space_highs, q_ct1);
    sycl::free(constMem.gpuG, q_ct1);
    sycl::free(constMem.cRuleWt, q_ct1);
    sycl::free(constMem.GPUScale, q_ct1);
    sycl::free(constMem.GPUNorm, q_ct1);
    sycl::free(constMem.gpuGenPos, q_ct1);
    sycl::free(constMem.gpuGenPermGIndex, q_ct1);
    sycl::free(constMem.gpuGenPermVarCount, q_ct1);
    sycl::free(constMem.gpuGenPermVarStart, q_ct1);
    sycl::free(constMem.cGeneratorCount, q_ct1);
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
    quad::cuda_memcpy_to_host<T>(h_generators,
                                 d_generators,
                                 ndim *
                                   pagani::CuhreFuncEvalsPerRegion<ndim>());

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

      quad::cuda_memcpy_to_host<double>(
        ests, estimates.integral_estimates, num_regions);
      quad::cuda_memcpy_to_host<double>(
        errs, estimates.error_estimates, num_regions);

      Print_region_evals(ests, errs, num_regions);

      if constexpr (debug >= 2) {
        quad::Func_Evals<ndim>* hfevals = new quad::Func_Evals<ndim>;
        hfevals->fevals_list = new quad::Feval<ndim>[num_regions * num_fevals];
        quad::cuda_memcpy_to_host<quad::Feval<ndim>>(
          hfevals->fevals_list, dfevals.fevals_list, num_regions * num_fevals);
        Print_func_evals(*hfevals, ests, errs, num_regions);
        delete[] hfevals->fevals_list;
        delete hfevals;
        sycl::free(dfevals.fevals_list, dpct::get_default_queue());
      }

      delete[] ests;
      delete[] errs;
    }
  }

  template <typename IntegT, int debug = 0>
  numint::integration_result
  apply_cubature_integration_rules(
    IntegT* d_integrand,
    int it,
    Sub_regs* subregions,
    Reg_estimates* subregion_estimates,
    Regs_characteristics* region_characteristics,
    bool compute_error = false)
  {

	//dpct::device_ext& dev_ct1 = dpct::get_current_device();
	//sycl::queue& q_ct1 = dev_ct1.default_queue();
	auto q_ct1 = sycl::queue(sycl::gpu_selector());
    size_t num_regions = subregions->size;
    quad::Func_Evals<ndim> dfevals;
	
    if constexpr (debug >= 2) {
      constexpr size_t num_fevals = pagani::CuhreFuncEvalsPerRegion<ndim>();
      dfevals.fevals_list =
        quad::cuda_malloc<quad::Feval<ndim>>(num_regions * num_fevals);
    }

    quad::set_device_array<int>(
      region_characteristics->active_regions, num_regions, 1.);

    size_t num_blocks = num_regions;
    constexpr size_t block_size = 64;

    T epsrel = 1.e-3, epsabs = 1.e-12;
	/*T* tempd = quad::cuda_malloc<T>(num_regions);//sycl::malloc_device<T>(num_regions, q_ct1);

	tempff(tempd);
	tempff(tempd);
	tempff(subregion_estimates->error_estimates);
	tempff(subregion_estimates->error_estimates);
	
	T* t = subregion_estimates->error_estimates;
	
	q_ct1.submit([&](sycl::handler& cgh) {
     cgh.parallel_for(
        sycl::nd_range(sycl::range(num_blocks * block_size),
                       sycl::range(block_size)),
        [=](sycl::nd_item<1> item_ct1) [[intel::reqd_sub_group_size(32)]] {
			
			subregion_estimates->error_estimates[0]=1.; //this doesn't workgroup
			//t[0] = 1.;
        });
    });
	//q_ct1.wait();
    q_ct1.wait_and_throw();
	std::cout<<"past manual"<<std::endl;*/
    //sycl::event start, stop;
    //std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    //std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    /*
    DPCT1026:153: The call to cudaEventCreate was removed because this call is
    redundant in DPC++.
    */
    /*
    DPCT1026:154: The call to cudaEventCreate was removed because this call is
    redundant in DPC++.
    */
    auto integral_estimates = subregion_estimates->integral_estimates;
    auto error_estimates = subregion_estimates->error_estimates;
    auto sub_dividing_dim = region_characteristics->sub_dividing_dim;
    auto dLeftCoord = subregions->dLeftCoord;
    auto dLength = subregions->dLength;
	
	q_ct1.submit([&](sycl::handler& cgh) {
      sycl::accessor<T /*Fix the type mannually*/,
                     1,
                     sycl::access_mode::read_write,
                     sycl::access::target::local>
        shared_acc_ct1(sycl::range(8), cgh);
      sycl::accessor<T,
                     1,
                     sycl::access_mode::read_write,
                     sycl::access::target::local>
        sdata_acc_ct1(sycl::range(block_size), cgh);
      sycl::accessor<T,
                     0,
                     sycl::access_mode::read_write,
                     sycl::access::target::local>
        Jacobian_acc_ct1(cgh);
      sycl::accessor<int,
                     0,
                     sycl::access_mode::read_write,
                     sycl::access::target::local>
        maxDim_acc_ct1(cgh);
      sycl::accessor<T,
                     0,
                     sycl::access_mode::read_write,
                     sycl::access::target::local>
        vol_acc_ct1(cgh);
      sycl::accessor<T,
                     1,
                     sycl::access_mode::read_write,
                     sycl::access::target::local>
        ranges_acc_ct1(sycl::range(ndim), cgh);
      sycl::accessor<Region<ndim>,
                     1,
                     sycl::access_mode::read_write,
                     sycl::access::target::local>
        sRegionPool_acc_ct1(sycl::range(1), cgh);
      sycl::accessor<GlobalBounds,
                     1,
                     sycl::access_mode::read_write,
                     sycl::access::target::local>
        sBound_acc_ct1(sycl::range(ndim), cgh);

      auto constMem_ct9 = constMem;
      auto integ_space_lows_ct10 = integ_space_lows;
      auto integ_space_highs_ct11 = integ_space_highs;
      auto generators_ct12 = generators;

      cgh.parallel_for(
        sycl::nd_range(sycl::range(num_blocks * block_size),
                       sycl::range(block_size)),
        [=](sycl::nd_item<1> item_ct1) [[intel::reqd_sub_group_size(32)]] {
			//tempd[0] = 1.;
			
          INTEGRATE_GPU_PHASE1<IntegT, T, ndim, block_size, debug>(
                d_integrand,
            dLeftCoord,
            dLength,
            num_regions,
            integral_estimates,
            error_estimates,
            sub_dividing_dim,
            epsrel,
            epsabs,
            constMem_ct9,
            integ_space_lows_ct10,
            integ_space_highs_ct11,
            generators_ct12,
            dfevals,
            item_ct1,
            shared_acc_ct1.get_pointer(),
            sdata_acc_ct1.get_pointer(),
            Jacobian_acc_ct1.get_pointer(),
            maxDim_acc_ct1.get_pointer(),
            vol_acc_ct1.get_pointer(),
            ranges_acc_ct1.get_pointer(),
            sRegionPool_acc_ct1.get_pointer(),
            sBound_acc_ct1.get_pointer());
        });
    });
	q_ct1.wait();
    //q_ct1.wait_and_throw();
    /*
    DPCT1012:157: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    //stop.wait();
    //stop_ct1 = std::chrono::steady_clock::now();
    //float kernel_time = 0;
    //kernel_time =
    //  std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    // std::cout<< "INTEGRATE_GPU_PHASE1-time:" << num_blocks << "," <<
    // kernel_time << std::endl;
    /*
    DPCT1007:158: Migration of this CUDA API is not supported by the Intel(R)
    DPC++ Compatibility Tool.
    */

    //print_verbose<debug>(generators, dfevals, subregion_estimates);

    numint::integration_result res;
    res.estimate = reduction<T, use_custom>(
      subregion_estimates->integral_estimates, num_regions);
    res.errorest = compute_error ?
                     reduction<T, use_custom>(
                       subregion_estimates->error_estimates, num_regions) :
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
    /*
    DPCT1049:159: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler& cgh) {
      auto generators_ct0 = generators;
      auto constMem_ct2 = constMem;

      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, block_size),
                                      sycl::range(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         ComputeGenerators<T, dim>(generators_ct0,
                                                   fEvalPerRegion,
                                                   constMem_ct2,
                                                   item_ct1);
                       });
    });
  }

  Structures<T> constMem;
  T* generators = nullptr;

  T* integ_space_lows = nullptr;
  T* integ_space_highs = nullptr;
};

template <typename T, size_t ndim, bool use_custom = false>
numint::integration_result
compute_finished_estimates(const Region_estimates<T, ndim>& estimates,
                           const Region_characteristics<ndim>& classifiers,
                           const numint::integration_result& iter)
{
  numint::integration_result finished;
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
accuracy_reached(T epsrel, T epsabs, numint::integration_result res)
{
  if (res.errorest / res.estimate <= epsrel || res.errorest <= epsabs)
    return true;
  return false;
}

template <typename T, typename IntegT, int ndim>
numint::integration_result
pagani_clone(const IntegT& integrand,
             Sub_regions<T, ndim>& subregions,
             T epsrel = 1.e-3,
             T epsabs = 1.e-12,
             bool relerr_classification = true)
{
  using Reg_estimates = Region_estimates<T, ndim>;
  using Sub_regs = Sub_regions<T, ndim>;
  using Regs_characteristics = Region_characteristics<ndim>;
  using Res = numint::integration_result;
  using Filter = Sub_regions_filter<T, ndim>;
  using Splitter = Sub_region_splitter<T, ndim>;
  Reg_estimates prev_iter_estimates;

  Res cummulative;
  Cubature_rules<T, ndim> cubature_rules;
  Heuristic_classifier<T, ndim> hs_classify(epsrel, epsabs);
  bool accuracy_termination = false;
  IntegT* d_integrand = quad::make_gpu_integrand<IntegT>(integrand);

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
