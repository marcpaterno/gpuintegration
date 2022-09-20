#ifndef ONE_API_WORKSPACE_H
#define ONE_API_WORKSPACE_H

#include "oneAPI/quad/Cubature_rules.h"
#include "oneAPI/quad/active_regions.h"
#include "oneAPI/quad/Finished_estimates.h"
#include "oneAPI/quad/Region_characteristics.h"
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/Sub_regions_filter.h"
#include "oneAPI/quad/Rule_Params.h"
#include "oneAPI/quad/heuristic_classifier.h"
#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/quad.h"
#include "oneAPI/quad/Classification_res.h"
#include "oneAPI/quad/Sub_region_splitter.h"
#include "oneAPI/quad/util/cuhreResult.h"
#include "oneAPI/quad/two_level_error_estimate.h"


template<size_t ndim>
class Workspace{
    using Estimates = Region_estimates<ndim>;
    using Sub_regs = Sub_regions<ndim>;
    using Reg_chars = Region_characteristics<ndim>;
    using Res = cuhreResult<double>;
    using Filter = Sub_regions_filter<ndim>;
    using Splitter = Sub_region_splitter<ndim>;
    using Classifier = Heuristic_classifier<ndim>;
    
    

    private:
    
        bool accuracy_reached(double epsrel, double epsabs, double estimate, double errorest){
            if(errorest/estimate <= epsrel || errorest <= epsabs)
                return true;
            return false;
        }
        
        void fix_error_budget_overflow(sycl::queue& q, Reg_chars& classifiers, const Res& finished, const Res& iter, Res& iter_finished, const double epsrel);
            
        bool heuristic_classify(sycl::queue& q, Classifier& classifier_a, Reg_chars& chars, const Estimates& ests, Res& finished, const Res& iter, const Res& cummulative);
        
        Cubature_rules<ndim> rules;
        sycl::queue* _q;    
        
    public:
        ~Workspace();
        Workspace(sycl::queue& q):rules(q){_q = &q;}
        
        template<typename F, int warp_size = 32, bool collect_iters = false, bool collect_sub_regions = false>
        Res integrate(sycl::queue& q, const F& integrand, double* lows, double* highs, Sub_regions<ndim>& subregions, double epsrel, double epsabs, bool relerr_classification = true);
};

template<size_t ndim>
Workspace<ndim>::~Workspace(){
}

template<size_t ndim>
bool
Workspace<ndim>::heuristic_classify(sycl::queue& q, 
    Classifier& classifier_a, 
    Region_characteristics<ndim>& chars, 
    const Estimates& estimates, 
    cuhreResult<double>& finished, 
    const cuhreResult<double>& iter, 
    const cuhreResult<double>& cummulative){
        
        using Classification = Classification_res;
        const size_t num_regions = chars.size;
		
        //const double total_mem = 16e9;
        const double needed_mem = static_cast<double>(classifier_a.device_mem_required_for_full_split(num_regions));
        const double ratio = needed_mem/static_cast<double>(free_device_mem(chars.size, ndim));
        
        const bool classification_necessary = ratio > 1.;
		//std::cout<<"free mem:"<<free_device_mem(chars.size, ndim) << std::endl;
		//std::cout<<"mem_needed:"<<classifier_a.device_mem_required_for_full_split(chars.size) << std::endl;
		//std::cout<<"ratio:"<< ratio<<std::endl;
        if(classifier_a.classification_criteria_met(q, num_regions) == false){
            const bool must_terminate = classification_necessary;
            return must_terminate;
        }
        
        Classification hs_results = 
	  classifier_a.classify(q, chars.active_regions, estimates.error_estimates, num_regions, iter.errorest, finished.errorest, cummulative.errorest);
        const bool success = hs_results.pass_mem && hs_results.pass_errorest_budget;
        
        if(success){
            free(chars.active_regions, q);
            chars.active_regions = hs_results.active_flags;
            finished.estimate = iter.estimate - quad::dot_product<double>(q, chars.active_regions, estimates.integral_estimates, num_regions);     
            finished.errorest = hs_results.finished_errorest;
        }    
        
        const bool must_terminate = (!success && classification_necessary) || hs_results.num_active == 0;
        return must_terminate;
}

template<size_t ndim>
void
Workspace<ndim>::fix_error_budget_overflow(sycl::queue& q, 
                                           Region_characteristics<ndim>& chars, 
                                           const cuhreResult<double>& cummulative_finished, 
                                           const cuhreResult<double>& iter, 
                                           cuhreResult<double>& iter_finished, 
                                           const double epsrel){
                                                
    double leaves_estimate = cummulative_finished.estimate + iter.estimate;
    double leaves_finished_errorest = cummulative_finished.errorest + iter_finished.errorest;
    const size_t num_regions = chars.size;
    
    if (leaves_finished_errorest > abs(leaves_estimate) * epsrel){
        size_t num_threads = 256;
        size_t num_blocks = chars.size / num_threads + ( num_regions % num_threads == 0 ? 0 : 1);
        double val = 1.;
        quad::parallel_fill<double>(q, chars.active_regions, num_regions, val);
        iter_finished.errorest = 0.;
        iter_finished.estimate = 0.;
    }
}

template<size_t ndim>
template<typename F, int warp_size, bool collect_iters, bool collect_sub_regions>
cuhreResult<double>       
  Workspace<ndim>::integrate(sycl::queue& q, const F& integrand, double* lows, double* highs, Sub_regions<ndim>& subregions, double epsrel, double epsabs, bool relerr_classification){
            using Res = cuhreResult<double>;
            
            Estimates prev_iter_estimates(q); 
            Res cummulative;
            Classifier classifier_a(epsrel, epsabs);
            cummulative.status = 1;
            bool compute_relerr_error_reduction = false;
            
            F* d_integrand = sycl::malloc_shared<F>(1,  q);
            memcpy(d_integrand, &integrand, sizeof(F));
            
            for(size_t it = 0; it < 700 && subregions.size > 0; it++){
                size_t num_regions = subregions.size;
                Reg_chars chars(q, subregions.size);
                Estimates estimates(q, subregions.size);

                Res iter = rules.template apply_cubature_integration_rules<F, warp_size>(q,
                d_integrand, lows, highs, epsrel, epsabs, subregions, estimates, chars, compute_relerr_error_reduction);
               
                two_level_errorest_and_relerr_classify<ndim>(q, estimates, prev_iter_estimates, chars, epsrel, relerr_classification);
				/*if(num_regions == 2066304){
					for(int i = 0 ; i < num_regions; ++i)
						printf("region_id %i, %f, %e, %e\n",  i, chars.active_regions[i], estimates.integral_estimates[i], estimates.error_estimates[i]);
				}*/
                iter.errorest = quad::reduction<double>(q, estimates.error_estimates, subregions.size);
				std::cout << it << "," << cummulative.estimate + iter.estimate << "," << cummulative.errorest + iter.errorest << "," << subregions.size  << std::endl;
			
                if(accuracy_reached(epsrel, epsabs, std::abs(cummulative.estimate + iter.estimate), cummulative.errorest + iter.errorest)){                    
                    free(d_integrand, q);
                    cummulative.estimate += iter.estimate;
                    cummulative.errorest += iter.errorest;
                    cummulative.status = 0;
                    cummulative.nregions += subregions.size;
                    
                    return cummulative;
                }
                
                
                classifier_a.store_estimate(cummulative.estimate + iter.estimate);
                Res finished = compute_finished_estimates<ndim>(q, estimates, chars, iter);          
                
                fix_error_budget_overflow(q, chars, cummulative, iter, finished, epsrel);
				

				
                if(heuristic_classify(q, classifier_a, chars, estimates, finished, iter, cummulative) == true){
                
                    cummulative.estimate += iter.estimate;
                    cummulative.errorest += iter.errorest;
                    cummulative.nregions += subregions.size;
                    free(d_integrand, q);
                    return cummulative;
                }               
                
                cummulative.estimate += finished.estimate;
                cummulative.errorest += finished.errorest; 
                Filter filter_obj(q, subregions.size);
                size_t num_active_regions = filter_obj.filter(q, subregions, chars, estimates, prev_iter_estimates);
                cummulative.nregions += num_regions - num_active_regions;
                subregions.size = num_active_regions;
                Splitter splitter(q, subregions.size);
                splitter.split(q, subregions, chars);
                cummulative.iters++;
            }
            
            cummulative.nregions += subregions.size;
            sycl::free(d_integrand, q);
            return cummulative; 
        }


#endif
