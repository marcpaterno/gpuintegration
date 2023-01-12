#ifndef HEURISTIC_CLASSIFIER_CUH
#define HEURISTIC_CLASSIFIER_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "pagani/quad/GPUquad/Sub_regions.dp.hpp"
#include "pagani/quad/util/mem_util.dp.hpp"
#include "pagani/quad/util/thrust_utils.dp.hpp"
#include <string>
#include <cmath>
#include <oneapi/mkl.hpp>
#include "oneapi/mkl/stats.hpp"

/*
	the dpct didn't have anything to do with this function at all,
	I took it from the manually developed version since dpct couldn't 
	do anything with the call to thrust::minmax_element
*/

/*template<typename T>
Range<T>
device_array_min_max(T* arr, size_t size){
	auto q = dpct::get_default_queue();
    Range<T> range;
    int64_t* min = sycl::malloc_shared<int64_t>(1, q);  
    int64_t* max = sycl::malloc_shared<int64_t>(1, q);    
    const int stride = 1;
	
	sycl::event est_ev = oneapi::mkl::blas::row_major::iamax(
		q, size, arr, stride, max);
					  
	sycl::event est_ev2 = oneapi::mkl::blas::row_major::iamin(
		q, size, arr, stride, min);
	
	est_ev.wait();
	est_ev2.wait();
	
    range.low = arr[min[0]];
    range.high = arr[max[0]];
    free(min, q);
    free(max, q);
    return range;
}*/

template<typename T>
Range<T>
device_array_min_max(T* arr, size_t size){
    Range<T> range;
	auto q = dpct::get_default_queue();
    double* min = sycl::malloc_shared<double>(1, q);  
    double* max = sycl::malloc_shared<double>(1, q);    
    
    oneapi::mkl::stats::dataset<oneapi::mkl::stats::layout::row_major, T*> wrapper(1, size, arr);
    
    auto this_event = oneapi::mkl::stats::min_max<oneapi::mkl::stats::method::fast, 
                        double, oneapi::mkl::stats::layout::row_major>(q, wrapper, min, max);
    this_event.wait();
    
    range.low = min[0];
    range.high = max[0];
    free(min, q);
    free(max, q);
	
	/*dpct::device_ext& dev_ct1 = dpct::get_current_device();
	sycl::queue& q_ct1 = dev_ct1.default_queue();
	dpct::device_pointer<T> d_ptr = dpct::get_device_pointer(arr);
	range.low = std::min_element (oneapi::dpl::execution::make_device_policy(q_ct1), arr, arr + size);*/
    return range;
}


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
        const double diff = fabs(threshold_range.low - threshold);
        threshold -= diff * .5;
    }

    void 
    increase_threshold(){
        const double diff = fabs(threshold_range.high - threshold);
        threshold += diff * .5;
    }
    
    bool pass_mem = false;
    bool pass_errorest_budget = false;
    
    double threshold = 0.;
    double errorest_budget_covered = 0.;
    double percent_mem_active = 0.;
    Range<double> threshold_range; //change to threshold_range
    double* active_flags = nullptr;
    size_t num_active = 0;
    double finished_errorest = 0.;
    
    double max_budget_perc_to_cover = .25;
    double max_active_perc = .5;
    bool data_allocated = false;
};

template <typename T>
void
device_set_true_for_larger_than(
         const T* arr,   
         const T val,
         const size_t size, 
         double* output_flags,
         sycl::nd_item<3> item_ct1)
{
    size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);

    if (tid < size) {                    
      output_flags[tid] = arr[tid] > val;
    }
}

template<typename T>
void set_true_for_larger_than(const T* arr, const T val, const size_t size, double* output_flags){
    size_t num_threads = 1024;
    size_t num_blocks = size/num_threads + (size % num_threads == 0 ? 0 : 1);
    /*
    DPCT1049:115: The workgroup size passed to the SYCL kernel may exceed
     * the limit. To get the device limit, query
     * info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    dpct::get_default_queue().parallel_for(
      sycl::nd_range(sycl::range(1, 1, num_blocks) *
                       sycl::range(1, 1, num_threads),
                     sycl::range(1, 1, num_threads)),
      [=](sycl::nd_item<3> item_ct1) {
          device_set_true_for_larger_than<T>(
            arr, val, size, output_flags, item_ct1);
      });
    dpct::get_current_device().queues_wait_and_throw();
}

size_t total_device_mem(){
    return dpct::get_current_device().get_device_info().get_global_mem_size();
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

size_t 
free_device_mem(size_t num_regions, size_t ndim){
	size_t total_physmem = total_device_mem();
    size_t mem_occupied = device_mem_required_for_full_split(num_regions, ndim);
	
	//the 1 is so we don't divide by zero at any point when using this
    size_t free_mem = total_physmem > mem_occupied ? total_physmem - mem_occupied : 1;
	return free_mem;
}



template<size_t ndim>
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
        
        size_t device_mem_required_for_full_split(size_t num_regions){
            return 8 * num_doubles_needed(num_regions) + 4 * num_ints_needed(num_regions);
        }
        
        bool  enough_mem_for_next_split(const size_t num_regions){
            return free_device_mem(num_regions, ndim) > device_mem_required_for_full_split(num_regions, ndim);
        }
                        
        bool need_further_classification(const size_t num_regions){
            if(estimate_converged() == false || enough_mem_for_next_split(num_regions, ndim) == true)
                return false;
            return true;  
        }            
        
        void 
        apply_threshold(Classification_res& res, const double* errorests, const size_t num_regions){
            
            auto int_division = [](int x, int y){
                return static_cast<double>(x)/static_cast<double>(y);
            };
            
            set_true_for_larger_than<double>(errorests, res.threshold, num_regions, res.active_flags);
            res.num_active  = static_cast<size_t>(reduction<double>(res.active_flags, num_regions));    
            res.percent_mem_active = int_division(res.num_active, num_regions);
            res.pass_mem = res.percent_mem_active <= max_active_regions_percentage;
        }
        
        void 
        evaluate_error_budget(Classification_res& res, 
            double* error_estimates,
            double* active_flags, 
            const size_t num_regions,
            const double target_error, 
            const double active_errorest, 
            const double iter_finished_errorest, 
            const double total_f_errorest,
            const double max_percent_err_budget){
            
            const double extra_f_errorest = active_errorest - dot_product<double, double>(error_estimates, active_flags, num_regions) - iter_finished_errorest;  
            const double error_budget = target_error - total_f_errorest;
            res.pass_errorest_budget = 
                extra_f_errorest <= max_percent_err_budget * error_budget;
            res.finished_errorest =  extra_f_errorest;           
        }
         
        void
        get_larger_threshold_results(Classification_res& thres_search, 
            const double* active_flags, 
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
            double ratio = static_cast<double>(device_mem_required_for_full_split(num_regions))/static_cast<double>(free_device_mem(num_regions, ndim));
					
            if(ratio > 1.)
                return true;
            else if(ratio > .1 && estimate_converged())
                return true;
            else
                return false;
        }
        
        Classification_res 
        classify(double* active_flags, //remove this param, it's unused
            double* errorests,
            const size_t num_regions,
            const double iter_errorest, 
            const double iter_finished_errorest,
            const double total_finished_errorest){
                           
            Classification_res thres_search = (device_array_min_max<double>(errorests, num_regions));
            thres_search.data_allocated = true;
            
            const double min_errorest = thres_search.threshold_range.low;
            const double max_errorest = thres_search.threshold_range.high;
            thres_search.threshold = iter_errorest/num_regions;
            thres_search.active_flags = cuda_malloc<double>(num_regions);
            const double target_error = abs(estimates_from_last_iters[2]) * epsrel;

            const size_t max_num_thresholds_attempts = 20;
            size_t num_thres_increases = 0;
            size_t num_thres_decreases = 0;
            size_t max_thres_increases = 20;
            
            int threshold_changed = 0; //keeps track of where the threshold is being pulled (left or right)

            do{
                
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
                sycl::free(thres_search.active_flags, dpct::get_default_queue());
            }
            
            thres_search.max_budget_perc_to_cover = max_percent_error_budget;
            thres_search.max_active_perc = max_active_regions_percentage;
            
            max_active_regions_percentage = .5;
            max_percent_error_budget = .25;
            
            return thres_search;
        }
};

#endif 