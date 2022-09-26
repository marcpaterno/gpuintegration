#ifndef WORKSPACE_CUH
#define WORKSPACE_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "oneAPI/pagani/quad/GPUquad/Region_estimates.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Sub_regions.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Region_characteristics.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/hybrid.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/PaganiUtils.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Sub_region_splitter.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/Sub_region_filter.dp.hpp"
#include "oneAPI/pagani/quad/GPUquad/heuristic_classifier.dp.hpp"
#include "oneAPI/pagani/quad/util/cuhreResult.dp.hpp"
#include "oneAPI/pagani/quad/util/Volume.dp.hpp"
#include <fstream>
#include <cmath>

template<bool debug_ters = false>
void output_iter_data(){
    if constexpr(!debug_ters)
        return;
}

template<size_t ndim, bool use_custom = false>
class Workspace{
    using Estimates = Region_estimates<ndim>;
    using Sub_regs = Sub_regions<ndim>;
    using Regs_characteristics = Region_characteristics<ndim>;
    using Res = cuhreResult<double>;
    using Filter = Sub_regions_filter<ndim, use_custom>;
    using Splitter = Sub_region_splitter<ndim>;
    using Classifier = Heuristic_classifier<ndim, use_custom>;
       
    private:
        void fix_error_budget_overflow(Region_characteristics<ndim>* classifiers, const cuhreResult<double>& finished, const cuhreResult<double>& iter, cuhreResult<double>& iter_finished, const double epsrel);
        bool heuristic_classify(Classifier& classifier_a, Regs_characteristics& characteristics, const Estimates& estimates, cuhreResult<double>& finished, const cuhreResult<double>& iter, const cuhreResult<double>& cummulative);
        
        Cubature_rules<ndim> rules;
                
    public:
        Workspace() = default;
        //Workspace(double* lows, double* highs):Cubature_rules<ndim>(lows, highs){}
        
        template<typename IntegT, bool debug>
        cuhreResult<double>
        integrate(const IntegT& integrand,
                           double epsrel,
                           double epsabs,
                           quad::Volume<double, ndim>& vol);
        
        template<typename IntegT, bool predict_split = false, bool collect_iters = false, bool collect_sub_regions = false, int debug = 0>
        cuhreResult<double> integrate(const IntegT& integrand, Sub_regions<ndim>& subregions, double epsrel, double epsabs, quad::Volume<double, ndim>& vol, bool relerr_classification = true);
};

template<size_t ndim, bool use_custom>
bool
Workspace<ndim, use_custom>::heuristic_classify(Classifier& classifier_a, 
    Region_characteristics<ndim>& characteristics, 
    const Estimates& estimates, 
    cuhreResult<double>& finished, 
    const Res& iter, 
    const cuhreResult<double>& cummulative){
    const double ratio = static_cast<double>(classifier_a.device_mem_required_for_full_split(characteristics.size))/static_cast<double>(free_device_mem(characteristics.size, ndim));    
    const bool classification_necessary = ratio > 1.;
    //std::cout<<"free mem:"<<free_device_mem(characteristics.size, ndim) << std::endl;
	//std::cout<<"mem_needed:"<<classifier_a.device_mem_required_for_full_split(characteristics.size) << std::endl;
	//std::cout<<"ratio:"<< ratio<<std::endl;
    if(!classifier_a.classification_criteria_met(characteristics.size)){
		//std::cout<<"classification criteria not met"<<std::endl;
        const bool must_terminate = classification_necessary;
        return must_terminate;
    }
	
    Classification_res hs_results = 
        classifier_a.classify(characteristics.active_regions, estimates.error_estimates, estimates.size, iter.errorest, finished.errorest, cummulative.errorest);
    const bool hs_classify_success = hs_results.pass_mem && hs_results.pass_errorest_budget;
	
    if(hs_classify_success){
        sycl::free(characteristics.active_regions, dpct::get_default_queue());
        characteristics.active_regions = hs_results.active_flags;
        finished.estimate = iter.estimate - dot_product<double, double, use_custom>(characteristics.active_regions, estimates.integral_estimates, characteristics.size);     
        finished.errorest = hs_results.finished_errorest;
    }    
        
    const bool must_terminate = (!hs_classify_success && classification_necessary) || hs_results.num_active == 0;
    return must_terminate;
}

template<size_t ndim, bool use_custom>
void
Workspace<ndim, use_custom>::fix_error_budget_overflow(Region_characteristics<ndim>* characteristics, 
                                            const cuhreResult<double>& cummulative_finished, 
                                            const cuhreResult<double>& iter, 
                                            cuhreResult<double>& iter_finished, 
                                            const double epsrel){
                                                
    double leaves_estimate = cummulative_finished.estimate + iter.estimate;
    double leaves_finished_errorest = cummulative_finished.errorest + iter_finished.errorest;

    if (leaves_finished_errorest > fabs(leaves_estimate) * epsrel) {
        size_t num_threads = 256;
        size_t num_blocks = characteristics->size / num_threads + ( characteristics->size % num_threads == 0 ? 0 : 1);
        /*
        DPCT1049:120: The workgroup size passed to the SYCL kernel
         * may exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
        */
        dpct::get_default_queue().parallel_for(
          sycl::nd_range(sycl::range(1, 1, num_blocks) *
                           sycl::range(1, 1, num_threads),
                         sycl::range(1, 1, num_threads)),
          [=](sycl::nd_item<3> item_ct1) {
              set_array_to_value<double>(characteristics->active_regions,
                                      characteristics->size,
                                      1,
                                      item_ct1);
          });
        dpct::get_current_device().queues_wait_and_throw();

        iter_finished.errorest = 0.;
        iter_finished.estimate = 0.;
    }
}

template <size_t ndim, bool use_custom>
template <typename IntegT,
          bool predict_split,
          bool collect_iters,
          bool collect_sub_regions,
          int debug>
cuhreResult<double>
Workspace<ndim, use_custom>::integrate(const IntegT& integrand,
                           Sub_regions<ndim>& subregions,
                           double epsrel,
                           double epsabs,
                           quad::Volume<double, ndim>& vol,
                           bool relerr_classification) {
			
            dpct::device_ext& dev_ct1 = dpct::get_current_device();
            sycl::queue& q_ct1 = dev_ct1.default_queue();

            Res cummulative;
            Recorder<debug> iter_recorder("oneapi_iters.csv"); 
            using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;

            rules.set_device_volume(vol.lows, vol.highs);
            Estimates prev_iter_estimates; 
            
            Classifier classifier_a(epsrel, epsabs);
            cummulative.status = 1;
            bool compute_relerr_error_reduction = false;
            IntegT* d_integrand = quad::cuda_copy_to_managed(integrand);
    
            if constexpr(debug > 0){
                iter_recorder.outfile << "it, estimate, errorest, nregions"<<std::endl;
            }
            
            for(size_t it = 0; it < 700 && subregions.size > 0; it++){
                size_t num_regions = subregions.size;
                Regs_characteristics characteristics(subregions.size);
                Estimates estimates(subregions.size);
                Res iter = rules.template apply_cubature_integration_rules<IntegT, collect_iters, debug>(d_integrand, it, &subregions, &estimates, &characteristics, compute_relerr_error_reduction);
				
                if(predict_split){
                    relerr_classification = subregions.size <= 15000000 && it < 15  && cummulative.nregions == 0 ? false : true;
                }
				
                two_level_errorest_and_relerr_classify<ndim>(&estimates, &prev_iter_estimates, &characteristics, epsrel, relerr_classification);
                iter.errorest = reduction<double, use_custom>(estimates.error_estimates, subregions.size);
                                       
                if(predict_split){
                    if(cummulative.nregions == 0 && it == 15){
                        subregions.take_snapshot();
                    }
                }
                
                 if constexpr(debug > 0)
                    iter_recorder.outfile << it << "," << cummulative.estimate + iter.estimate << "," << cummulative.errorest + iter.errorest << "," << subregions.size  << std::endl;
                std::cout<< it << "," << cummulative.estimate + iter.estimate << "," << cummulative.errorest + iter.errorest << "," << subregions.size  << std::endl;
				if(accuracy_reached(epsrel, epsabs, std::abs(cummulative.estimate + iter.estimate), cummulative.errorest + iter.errorest)){
                    
                    cummulative.estimate += iter.estimate;
                    cummulative.errorest += iter.errorest;
                    cummulative.status = 0;
                    cummulative.nregions += subregions.size;
                    d_integrand->~IntegT();
                    sycl::free(d_integrand, q_ct1);
                    dpct::get_current_device().queues_wait_and_throw(); //is this what stopped the error? REALLY?
                    std::cout<<"total_time:"<<rules.total_time/1e6 << std::endl;
                    return cummulative;
                }
				
                classifier_a.store_estimate(cummulative.estimate + iter.estimate);
                Res finished = compute_finished_estimates<ndim>(estimates, characteristics, iter); 
                fix_error_budget_overflow(&characteristics, cummulative, iter, finished, epsrel);
                dpct::get_current_device().queues_wait_and_throw();
				
                if(heuristic_classify(classifier_a, characteristics, estimates, finished, iter, cummulative) == true){
                    cummulative.estimate += iter.estimate;
                    cummulative.errorest += iter.errorest;
                    cummulative.nregions += subregions.size;
                    d_integrand->~IntegT();
                    sycl::free(d_integrand, q_ct1);
                    return cummulative;
                }

                cummulative.estimate += finished.estimate;
                cummulative.errorest += finished.errorest; 

                Filter filter_obj(subregions.size);
                size_t num_active_regions = filter_obj.filter(&subregions, &characteristics, &estimates, &prev_iter_estimates);
                cummulative.nregions += num_regions - num_active_regions;
                subregions.size = num_active_regions;

                Splitter splitter(subregions.size);
                splitter.split(&subregions, &characteristics);
                cummulative.iters++;
            }
            std::cout<<"total_time:"<<rules.total_time << std::endl;
            d_integrand->~IntegT();
            cummulative.nregions += subregions.size;
            sycl::free(d_integrand, q_ct1);
            return cummulative;
            
        }

template <size_t ndim, bool use_custom>
template <typename IntegT, bool debug>
cuhreResult<double>
Workspace<ndim, use_custom>::integrate(const IntegT& integrand,
                           double epsrel,
                           double epsabs,
                           quad::Volume<double, ndim>& vol)
{
    bool relerr_classification = true;
    size_t partitions_per_axis = 2;   
    if(ndim < 5)
        partitions_per_axis = 4;
    else if(ndim <= 10)
        partitions_per_axis = 2;
    else
        partitions_per_axis = 1;

    Sub_regions<ndim> sub_regions(partitions_per_axis);
    sub_regions.uniform_split(partitions_per_axis);
     
     constexpr bool predict_split = false;
     constexpr bool collect_iters = false;
     constexpr bool collect_sub_regions = false;
    
    cuhreResult<double> result = integrate<IntegT, predict_split, collect_iters, collect_sub_regions, debug>
        (integrand, sub_regions, epsrel, epsabs, vol, relerr_classification);
    return result;
}

#endif