#ifndef HEURISTIC_CLASSIFIER_CUH
#define HEURISTIC_CLASSIFIER_CUH

#include "cuda/pagani/quad/GPUquad/Sub_regions.cuh"
#include "cuda/pagani/quad/util/mem_util.cuh"
#include "cuda/pagani/quad/util/thrust_utils.cuh"
#include "cuda/pagani/quad/util/custom_functions.cuh"

#include <string>

 std::string
 doubleToString(double val, int prec_level){
      std::ostringstream out;
      out.precision(prec_level);
      out << std::fixed << val;
      return out.str();
 }

//needs renaming

class Classification_res{
    public:
    Classification_res() = default;
    Classification_res(Range<double> some_range):threshold_range(some_range){}
    
    ~Classification_res(){}
    
    void 
    decrease_threshold(){
        const double diff = abs(threshold_range.low - threshold);
        threshold -= diff * .5;
    }

    void 
    increase_threshold(){
        const double diff = abs(threshold_range.high - threshold);
        threshold += diff * .5;
    }
    
    bool pass_mem = false;
    bool pass_errorest_budget = false;
    
    double threshold = 0.;
    double errorest_budget_covered = 0.;
    double percent_mem_active = 0.;
    Range<double> threshold_range; //change to threshold_range
    int* active_flags = nullptr;
    size_t num_active = 0;
    double finished_errorest = 0.;
    
    double max_budget_perc_to_cover = .25;
    double max_active_perc = .5;
    bool data_allocated = false;
};

template <typename T>
__global__ void
device_set_true_for_larger_than(
         const T* arr,   
         const T val,
         const size_t size, 
         int* output_flags)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {                    
      //printf("setting active[%lu]: %i becuase arr[%lu]:%f val:%f\n", tid, arr[tid] > val, tid, arr[tid], val);
      output_flags[tid] = arr[tid] > val;
    }
}

template<typename T>
void set_true_for_larger_than(const T* arr, const T val, const size_t size, int* output_flags){
    size_t num_threads = 1024;
    size_t num_blocks = size/num_threads + (size % num_threads == 0 ? 0 : 1);
    device_set_true_for_larger_than<T><<<num_blocks, num_threads>>>(arr, val, size, output_flags);
    cudaDeviceSynchronize();
    quad::CudaCheckError();
}

size_t total_device_mem(){
    //return dpct::get_current_device().get_device_info().get_global_mem_size();
	return 16e9; //ONLY FOR CUDA_BACKEND maybe adjust with a template argument?
}

size_t  
num_ints_needed(size_t num_regions){//move to pagani utils, has nothing to do with classifying
    const size_t scanned = num_regions;
    const size_t subDivDim = 2 * num_regions;
    const size_t activeBisectDim = num_regions;
    return activeBisectDim + subDivDim + scanned;
}

size_t  
num_doubles_needed(size_t num_regions, size_t ndim){//move to pagani utils, has nothing to do with classifying
    const size_t newActiveRegions = num_regions * ndim;
    const size_t newActiveRegionsLength = num_regions * ndim;
    const size_t parentExpansionEstimate = num_regions;
    const size_t parentExpansionErrorest = num_regions;
    const size_t genRegions = num_regions * ndim * 2;
    const size_t genRegionsLength = num_regions * ndim * 2;
            
    const size_t regions = 2 * num_regions * ndim;
    const size_t regionsLength = 2 * num_regions * ndim;
    const size_t regionsIntegral = 2 * num_regions;
    const size_t regionsError = 2 * num_regions;
    const size_t parentsIntegral = num_regions;
    const size_t parentsError = num_regions;
            
    return parentsError + parentsIntegral+regionsError + regionsIntegral + regionsLength + regions + genRegionsLength + genRegions+parentExpansionErrorest + parentExpansionEstimate+newActiveRegionsLength + newActiveRegions;
}

size_t 
device_mem_required_for_full_split(size_t num_regions, size_t ndim){
    return 8 * num_doubles_needed(num_regions, ndim) + 4 * num_ints_needed(num_regions);
}

/*
size_t free_device_mem(){
    size_t free_physmem, total_physmem;
    cudaMemGetInfo(&free_physmem, &total_physmem);
    return free_physmem;
}*/

size_t 
free_device_mem(size_t num_regions, size_t ndim){
	size_t total_physmem = total_device_mem();
    size_t mem_occupied = device_mem_required_for_full_split(num_regions, ndim);
	
	//the 1 is so we don't divide by zero at any point when using this
    size_t free_mem = total_physmem > mem_occupied ? total_physmem - mem_occupied : 1;
	return free_mem;
}

template<size_t ndim, bool use_custom = false>
class Heuristic_classifier{
    
     double epsrel = 0.;
     double epsabs = 0.;
     int required_digits = 0;
     std::array<double, 3> estimates_from_last_iters;
     size_t iters_collected = 0;
     const size_t min_iters_for_convergence = 1;
     double max_percent_error_budget = .25;
     double max_active_regions_percentage = .5;
     
     friend class Classification_res;
     
    public:
		
        Heuristic_classifier() = default;
        
        Heuristic_classifier(double rel_tol, double abs_tol):epsrel(rel_tol), epsabs(abs_tol){
            required_digits = ceil(log10(1 / epsrel));
        }
          
        bool
        sigDigitsSame()
        {
		  //std::cout<<"required_digits:"<<required_digits<<std::endl;
          double third = abs(estimates_from_last_iters[0]);
          double second = abs(estimates_from_last_iters[1]);
          double first = abs(estimates_from_last_iters[2]);

          while (first != 0. && first < 1.) {
            first *= 10;
          }
          while (second != 0. && second < 1.) {
            second *= 10;
          }
          while (third != 0. && third < 1.) {
            third *= 10;
          }
        
          std::string second_to_last = doubleToString(third, 15);
          std::string last = doubleToString(second, 15);
          std::string current = doubleToString(first, 15);

          bool verdict = true;
          int sigDigits = 0;
          
          for (int i = 0; i < required_digits + 1 && sigDigits < required_digits &&
                          verdict == true;
               ++i) {
            verdict =
              current[i] == last[i] && last[i] == second_to_last[i] ? true : false;

            sigDigits += (verdict == true && current[i] != '.') ? 1 : 0;
          }
          return verdict;
        }
        
        bool estimate_converged(){
            //the -1 is because iters_collected++ is 1 at iteration 0 and I don't want to start counting at -1
            //printf("min_iters_for_convergence for convergenc:%lu\n", min_iters_for_convergence);
            if(iters_collected - 1 < min_iters_for_convergence || !sigDigitsSame())
                return false;
			else{
				return true;
			}
        }
        
        void store_estimate(const double estimate){
            estimates_from_last_iters[0] = estimates_from_last_iters[1];
            estimates_from_last_iters[1] = estimates_from_last_iters[2];
            estimates_from_last_iters[2] = estimate;
            iters_collected++;
        }
        
        size_t num_doubles_needed(const size_t num_regions){//move to pagani utils, has nothing to do with classifying
            const size_t newActiveRegions = num_regions * ndim;
            const size_t newActiveRegionsLength = num_regions * ndim;
            const size_t parentExpansionEstimate = num_regions;
            const size_t parentExpansionErrorest = num_regions;
            const size_t genRegions = num_regions * ndim * 2;
            const size_t genRegionsLength = num_regions * ndim * 2;
            
            const size_t regions = 2 * num_regions * ndim;
            const size_t regionsLength = 2 * num_regions * ndim;
            const size_t regionsIntegral = 2 * num_regions;
            const size_t regionsError = 2 * num_regions;
            const size_t parentsIntegral = num_regions;
            const size_t parentsError = num_regions;
            
            return parentsError + parentsIntegral+regionsError + regionsIntegral + regionsLength + regions + genRegionsLength + genRegions+parentExpansionErrorest + parentExpansionEstimate +newActiveRegionsLength + newActiveRegions;
        }
        
        size_t num_ints_needed(const size_t num_regions){//move to pagani utils, has nothing to do with classifying
            const size_t scanned = num_regions;
            const size_t subDivDim = 2 * num_regions;
            const size_t activeBisectDim = num_regions;
            return activeBisectDim + subDivDim + scanned;
        }
        
        size_t device_mem_required_for_full_split(const size_t num_regions){
            return 8 * num_doubles_needed(num_regions) + 4 * num_ints_needed(num_regions);
        }
        
        bool  enough_mem_for_next_split(const size_t num_regions){
            return free_device_mem(num_regions, ndim) > device_mem_required_for_full_split(num_regions);
        }
                        
        bool need_further_classification(const size_t num_regions){
            if(estimate_converged() == false || enough_mem_for_next_split(num_regions) == true)
                return false;
            return true;  
        }            
        
        void 
        apply_threshold(Classification_res& res, const double* errorests, const size_t num_regions){
            
            auto int_division = [](int x, int y){
                return static_cast<double>(x)/static_cast<double>(y);
            };
            
            set_true_for_larger_than<double>(errorests, res.threshold, num_regions, res.active_flags);
            res.num_active  = static_cast<size_t>(reduction<int, use_custom>(res.active_flags, num_regions));    
            res.percent_mem_active = int_division(res.num_active, num_regions);
			//std::cout<<"res.num_active:"<< res.num_active << std::endl;
			//std::cout<<"res.percent_mem_active:"<<res.percent_mem_active<<std::endl;
			//std::cout<<"max_active_regions_percentage:"<<max_active_regions_percentage<<std::endl;
            res.pass_mem = res.percent_mem_active <= max_active_regions_percentage;
			//std::cout<<"heuristic classifier reduction (num_active_regions:)"<< res.num_active << std::endl;

        }
        
        void 
        evaluate_error_budget(Classification_res& res, 
            double* error_estimates,
            int* active_flags, 
            const size_t num_regions,
            const double target_error, 
            const double active_errorest, 
            const double iter_finished_errorest, 
            const double total_f_errorest,
            const double max_percent_err_budget){
            
            const double extra_f_errorest = active_errorest - dot_product<double, int, use_custom>(error_estimates, active_flags, num_regions) - iter_finished_errorest;  
            //std::cout<<"dot_product<double, double, use_custom>(error_estimates, active_flags, num_regions):"<<dot_product<double, int, use_custom>(error_estimates, active_flags, num_regions)<<std::endl;
			//std::cout<<"extra_f_errorest:"<<extra_f_errorest<<std::endl;
			//std::cout<<"iter_finished_errorest:"<<iter_finished_errorest<<std::endl;
			const double error_budget = target_error - total_f_errorest;
            res.pass_errorest_budget = 
                extra_f_errorest <= max_percent_err_budget * error_budget;
            res.finished_errorest =  extra_f_errorest;           
        }
         
        void
        get_larger_threshold_results(Classification_res& thres_search, 
            const int* active_flags, 
            const double* errorests, 
            const size_t num_regions){
                
            thres_search.pass_mem = false;
            const size_t max_attempts = 20;
            size_t counter = 0;
            
            while(!thres_search.pass_mem && counter < max_attempts){
                apply_threshold(thres_search, errorests, num_regions);
                if(!thres_search.pass_mem){
                    thres_search.threshold_range.low = thres_search.threshold;
                    thres_search.increase_threshold();   
                }
                counter++;
            }
        }
        
        bool 
		classification_criteria_met(const size_t num_regions){
            double ratio = static_cast<double>(device_mem_required_for_full_split(num_regions))/static_cast<double>(/*quad::GetAmountFreeMem()*/free_device_mem(num_regions, ndim));
					
            if(ratio > 1.)
                return true;
            else if(ratio > .1 && estimate_converged())
                return true;
            else
                return false;
        }
        
        Classification_res 
        classify(int* active_flags, //remove this param, it's unused
            double* errorests,
            const size_t num_regions,
            const double iter_errorest, 
            const double iter_finished_errorest,
            const double total_finished_errorest){
                           
            Classification_res thres_search = (device_array_min_max<double, use_custom>(errorests, num_regions));
            thres_search.data_allocated = true;
            
            const double min_errorest = thres_search.threshold_range.low;
            const double max_errorest = thres_search.threshold_range.high;
            thres_search.threshold = iter_errorest/num_regions;
            thres_search.active_flags = cuda_malloc<int>(num_regions);
            const double target_error = abs(estimates_from_last_iters[2]) * epsrel;

            const size_t max_num_thresholds_attempts = 20;
            size_t num_thres_increases = 0;
            size_t num_thres_decreases = 0;
            size_t max_thres_increases = 20;
            
            int threshold_changed = 0; //keeps track of where the threshold is being pulled (left or right)
			//std::cout << "pass_mem:"<< thres_search.pass_mem << std::endl;
			//std::cout<< "pass_error_budget:"<< thres_search.pass_errorest_budget << std::endl;
			
            do{
                std::cout<<"classifying"<<std::endl;
                if(!thres_search.pass_mem && num_thres_increases <= max_thres_increases){
                    get_larger_threshold_results(thres_search, active_flags, errorests, num_regions);
                    num_thres_increases++;
                }
                        
                if(thres_search.pass_mem){
                    evaluate_error_budget(thres_search, 
                        errorests, 
                        thres_search.active_flags, 
                        num_regions, 
                        target_error, 
                        iter_errorest, 
                        iter_finished_errorest, 
                        total_finished_errorest,
                        max_percent_error_budget);
                    
                    if(!thres_search.pass_errorest_budget && num_thres_decreases <= max_num_thresholds_attempts){
                        thres_search.threshold_range.high = thres_search.threshold;
                        thres_search.decrease_threshold();
                        thres_search.pass_mem = false;   //we don't know if it will pass
                        num_thres_decreases++;
                        threshold_changed++;
                    }
                }
                
                bool exhausted_attempts = num_thres_decreases >= 20  || num_thres_increases >= 20;
                
                if(exhausted_attempts && max_percent_error_budget < .7){
                    max_percent_error_budget += 0.1;
                    num_thres_decreases = 0;
                    num_thres_increases = 0;
                    thres_search.threshold_range.low = min_errorest;
                    thres_search.threshold_range.high = max_errorest;
                    thres_search.threshold = iter_errorest/num_regions;
                }else if(exhausted_attempts && max_percent_error_budget >= .7 && max_active_regions_percentage <= .7){
                    max_active_regions_percentage += .1;
                }
                else if(exhausted_attempts){   
                    break;
                }
            }while(!thres_search.pass_mem ||  !thres_search.pass_errorest_budget);
            
            if(!thres_search.pass_mem || !thres_search.pass_errorest_budget){
                cudaFree(thres_search.active_flags);
            }
            
            thres_search.max_budget_perc_to_cover = max_percent_error_budget;
            thres_search.max_active_perc = max_active_regions_percentage;
            
            max_active_regions_percentage = .5;
            max_percent_error_budget = .25;
            
            return thres_search;
        }
};

#endif 