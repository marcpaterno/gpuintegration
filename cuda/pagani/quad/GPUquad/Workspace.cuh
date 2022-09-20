#ifndef WORKSPACE_CUH
#define WORKSPACE_CUH

#include "cuda/pagani/quad/GPUquad/Region_estimates.cuh"
#include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "cuda/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "cuda/pagani/quad/GPUquad/hybrid.cuh"
#include "cuda/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "cuda/pagani/quad/GPUquad/Sub_region_splitter.cuh"
#include "cuda/pagani/quad/GPUquad/Sub_region_filter.cuh"
#include "cuda/pagani/quad/GPUquad/heuristic_classifier.cuh"
#include "cuda/pagani/quad/util/cuhreResult.cuh"
#include "cuda/pagani/quad/util/Volume.cuh"

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
    using Classifier = Heuristic_classifier<ndim>;
    std::ofstream outiters;
	
    private:
        void fix_error_budget_overflow(Region_characteristics<ndim>& classifiers, const cuhreResult<double>& finished, const cuhreResult<double>& iter, cuhreResult<double>& iter_finished, const double epsrel);
        bool heuristic_classify(Classifier& classifier_a, Regs_characteristics& characteristics, const Estimates& estimates, cuhreResult<double>& finished, const cuhreResult<double>& iter, const cuhreResult<double>& cummulative);
        
        Cubature_rules<ndim> rules;
                
    public:
        Workspace() = default;
		Workspace(double* lows, double* highs):Cubature_rules<ndim>(lows, highs){}
        template<typename IntegT, bool predict_split = false, bool collect_iters = false, int debug = 0>
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
        
        const double ratio = static_cast<double>(classifier_a.device_mem_required_for_full_split(characteristics.size))/static_cast<double>(free_device_mem(characteristics.size, ndim)/*quad::GetAmountFreeMem()*/);
        const bool classification_necessary = ratio > 1.;
		//std::cout<<"free mem:"<<free_device_mem(characteristics.size, ndim) << std::endl;
		//std::cout<<"mem_needed:"<<classifier_a.device_mem_required_for_full_split(characteristics.size) << std::endl;
		//std::cout<<"ratio:"<< ratio<<std::endl;
        if(!classifier_a.classification_criteria_met(characteristics.size)){
            const bool must_terminate = classification_necessary;
            return must_terminate;
        }
        
        Classification_res hs_results = 
                    classifier_a.classify(characteristics.active_regions, estimates.error_estimates, estimates.size, iter.errorest, finished.errorest, cummulative.errorest);
        const bool hs_classify_success = hs_results.pass_mem && hs_results.pass_errorest_budget;
        
        if(hs_classify_success){
            cudaFree(characteristics.active_regions);
            characteristics.active_regions = hs_results.active_flags;
            finished.estimate = iter.estimate - dot_product<int, double, use_custom>(characteristics.active_regions, estimates.integral_estimates, characteristics.size);     
            finished.errorest = hs_results.finished_errorest;
        }    
        
        const bool must_terminate = (!hs_classify_success && classification_necessary) || hs_results.num_active == 0;
        return must_terminate;
}

template<size_t ndim, bool use_custom>
void
Workspace<ndim, use_custom>::fix_error_budget_overflow(Region_characteristics<ndim>& characteristics, 
                                            const cuhreResult<double>& cummulative_finished, 
                                            const cuhreResult<double>& iter, 
                                            cuhreResult<double>& iter_finished, 
                                            const double epsrel){
                                                
    double leaves_estimate = cummulative_finished.estimate + iter.estimate;
    double leaves_finished_errorest = cummulative_finished.errorest + iter_finished.errorest;
    
    if (leaves_finished_errorest > abs(leaves_estimate) * epsrel){
        size_t num_threads = 256;
        size_t num_blocks = characteristics.size / num_threads + ( characteristics.size % num_threads == 0 ? 0 : 1);
        set_array_to_value<int><<<num_blocks, num_threads>>>(characteristics.active_regions, characteristics.size, 1);
        cudaDeviceSynchronize();
        
        iter_finished.errorest = 0.;
        iter_finished.estimate = 0.;
    }
}

template<size_t ndim, bool use_custom>
template<typename IntegT, bool predict_split, bool collect_iters, int debug>
cuhreResult<double>       
Workspace<ndim, use_custom>::integrate(const IntegT& integrand, Sub_regions<ndim>& subregions, double epsrel, double epsabs, quad::Volume<double, ndim>& vol, bool relerr_classification){
            using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;

			rules.set_device_volume(vol.lows, vol.highs);
            Estimates prev_iter_estimates; 
            Res cummulative;
			Recorder<debug> iter_recorder("cuda_iters.csv"); 

            Classifier classifier_a(epsrel, epsabs);
            cummulative.status = 1;
            bool compute_relerr_error_reduction = false;
            
            IntegT* d_integrand = make_gpu_integrand<IntegT>(integrand);
			
			if constexpr(debug > 0){
				
				iter_recorder.outfile << "it, estimate, errorest, nregions"<<std::endl;
            }
			
            for(size_t it = 0; it < 700 && subregions.size > 0; it++){
                //std::cout<<" start of iteration quad::GetAmountFreeMem():"<<quad::GetAmountFreeMem()<<std::endl;
                size_t num_regions = subregions.size;
                Regs_characteristics characteristics(subregions.size);
                Estimates estimates(subregions.size);
				
                auto const t0 = std::chrono::high_resolution_clock::now();
                Res iter = rules.template apply_cubature_integration_rules<IntegT, debug>(d_integrand, it, subregions, estimates, characteristics, compute_relerr_error_reduction);
                MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
                
				if constexpr(predict_split){
					relerr_classification = subregions.size <= 15000000 && it < 15  && cummulative.nregions == 0 ? false : true;
				}
                two_level_errorest_and_relerr_classify<ndim>(estimates, prev_iter_estimates, characteristics, epsrel, relerr_classification);
                
                iter.errorest = reduction<double, use_custom>(estimates.error_estimates, subregions.size);
                                       
				 if constexpr(debug > 0)
                    iter_recorder.outfile << it << "," << cummulative.estimate + iter.estimate << "," << cummulative.errorest + iter.errorest << "," << subregions.size  << std::endl;
				 std::cout << it << "," << cummulative.estimate + iter.estimate << "," << cummulative.errorest + iter.errorest << "," << subregions.size  << std::endl;

                if(predict_split){
					if(cummulative.nregions == 0 && it == 15 /*&& subregions.size <= (size_t)(pow((double)2, double(ndim+20)))*/){
						subregions.take_snapshot();
					}
				}
				
                if(it == 17){
				//if(accuracy_reached(epsrel, epsabs, std::abs(cummulative.estimate + iter.estimate), cummulative.errorest + iter.errorest)){
                    cummulative.estimate += iter.estimate;
                    cummulative.errorest += iter.errorest;
                    cummulative.status = 0;
                    cummulative.nregions += subregions.size;
                    cudaFree(d_integrand);
                    return cummulative;
                }
                
                quad::CudaCheckError();
                classifier_a.store_estimate(cummulative.estimate + iter.estimate);
				
                Res finished = compute_finished_estimates<ndim, use_custom>(estimates, characteristics, iter);   
                fix_error_budget_overflow(characteristics, cummulative, iter, finished, epsrel);
                if(heuristic_classify(classifier_a, characteristics, estimates, finished, iter, cummulative) == true){
                    cummulative.estimate += iter.estimate;
                    cummulative.errorest += iter.errorest;
                    cummulative.nregions += subregions.size;
                    cudaFree(d_integrand);
                    return cummulative;
                }               

                cummulative.estimate += finished.estimate;
                cummulative.errorest += finished.errorest; 
                                
                Filter filter_obj(subregions.size);
                size_t num_active_regions = filter_obj.filter(subregions, characteristics, estimates, prev_iter_estimates);
                cummulative.nregions += num_regions - num_active_regions;
                subregions.size = num_active_regions;
                  
                Splitter splitter(subregions.size);
                splitter.split(subregions, characteristics);
                cummulative.iters++;
            }
            cummulative.nregions += subregions.size;
            cudaFree(d_integrand);
            return cummulative;
            
        }


#endif