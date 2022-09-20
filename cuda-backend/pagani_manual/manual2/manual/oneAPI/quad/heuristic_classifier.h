#ifndef ONE_API_HEURISTIC_CLASSIFIER_H
#define ONE_API_HEURISTIC_CLASSIFIER_H

#include "oneAPI/quad/util/MemoryUtil.h"
#include "oneAPI/quad/Classification_res.h"
#include <CL/sycl.hpp>
//#include "oneapi/mkl.hpp"
//#include "oneapi/mkl/stats.hpp"

template<typename T>
void
cuda_memcpy_to_host(sycl::queue& q, T* dest, T* src, size_t size){
    q.memcpy(dest, src, sizeof(T) * size).wait();
}


template<typename T>
quad::Range<T>
device_array_min_max(sycl::queue& q, T* arr, size_t size){
    quad::Range<T> range;
    /*double* min = sycl::malloc_shared<double>(1, q);  
    double* max = sycl::malloc_shared<double>(1, q);    
    
    oneapi::mkl::stats::dataset<oneapi::mkl::stats::layout::row_major, T*> wrapper(1, size, arr);
    
    auto this_event = oneapi::mkl::stats::min_max<oneapi::mkl::stats::method::fast, 
                        double, oneapi::mkl::stats::layout::row_major>(q, wrapper, min, max);
    this_event.wait();
    
    range.low = min[0];
    range.high = max[0];
    free(min, q);
    free(max, q);
    return range;*/
    //int64_t* dmin = sycl::malloc_device<int64_t>(1, q);  
	//int64_t* dmax = sycl::malloc_device<int64_t>(1, q);    
	const int stride = 1;
	int64_t* dmin = new int64_t;  
	int64_t* dmax = new int64_t;  		
	sycl::event est_ev = oneapi::mkl::blas::column_major::iamax(
			q, size, arr, stride, dmax);
						  
	sycl::event est_ev2 = oneapi::mkl::blas::column_major::iamin(
			q, size, arr, stride, dmin);
	est_ev.wait();
	est_ev2.wait();
	
	int64_t* maxi = quad::allocate_and_copy_to_host<int64_t>(dmax, 1);; 
	int64_t* mini = quad::allocate_and_copy_to_host<int64_t>(dmin, 1);

	T* maxv = quad::allocate_and_copy_to_host<T>(&arr[*maxi], 1);; 
	T* minv = quad::allocate_and_copy_to_host<T>(&arr[*mini], 1);
	
	range.low = *minv;
	range.high = *maxv;;
	return range;
}

std::string
doubleToString(double val, int prec_level){
      std::ostringstream out;
      out.precision(prec_level);
      out << std::fixed << val;
      return out.str();
}

template<typename T>
void set_true_for_larger_than(sycl::queue& q, const T* arr, const T val, const size_t size, double* output_flags){  
    q.submit([&](auto &cgh) {
        cgh.parallel_for(sycl::range<1>(size),
               [=](sycl::id<1> gtid){    
             output_flags[gtid] = static_cast<double>(arr[gtid] > val); //since this is a double it needs a cast, TODO in cuda
         });    
    }).wait(); 
}

size_t total_device_mem(){
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
     const size_t min_iters_for_convergence = 10;
     double max_percent_error_budget = .25;
     double max_active_regions_percentage = .5;
     
     friend class Classification_res;
     
    public:
       
        Heuristic_classifier() = default;
        
        Heuristic_classifier(double rel_tol, double abs_tol):epsrel(rel_tol), epsabs(abs_tol){
            required_digits = ceil(log10(1 / epsrel));
        }
          
        bool
        sigDigitsSame();
        
        bool estimate_converged();
        
        void store_estimate(const double estimate);
        
        size_t num_doubles_needed(const size_t num_regions);
        
        size_t 
        num_ints_needed(const size_t num_regions);
        
        size_t 
        device_mem_required_for_full_split(const size_t num_regions);
        
        bool  
        enough_mem_for_next_split(sycl::queue& q, const size_t num_regions);
                        
        bool 
        need_further_classification(sycl::queue& q, const size_t num_regions);
        
        void 
        apply_threshold(sycl::queue& q, Classification_res& res, const double* errorests, const size_t num_regions);
        
        void 
        evaluate_error_budget(sycl::queue& q, Classification_res& res, double* error_estimates, double* active_flags, const size_t num_regions, const double target_error, const double active_errorest, const double iter_finished_errorest, const double total_f_errorest, const double max_percent_err_budget);
         
        void
        get_larger_threshold_results(sycl::queue& q, Classification_res& thres_search, const double* active_flags, const double* errorests, const size_t num_regions);
        
        bool classification_criteria_met(sycl::queue& q, const size_t num_regions);
        
        Classification_res 
        classify(sycl::queue& q, double* active_flags, double* errorests, const size_t num_regions, const double iter_errorest, const double iter_finished_errorest, const double total_finished_errorest);
};

template<size_t ndim>
bool Heuristic_classifier<ndim>::need_further_classification(sycl::queue& q, const size_t num_regions){
    if(estimate_converged() == false || enough_mem_for_next_split(q, num_regions) == true)
        return false;
    return true;  
}   

template<size_t ndim>
bool  
Heuristic_classifier<ndim>::enough_mem_for_next_split(sycl::queue& q, const size_t num_regions){
    return quad::total_device_mem(q, num_regions) > device_mem_required_for_full_split(num_regions);
}

template<size_t ndim>
size_t 
Heuristic_classifier<ndim>::device_mem_required_for_full_split(const size_t num_regions){
    return 8 * num_doubles_needed(num_regions) + 4 * num_ints_needed(num_regions);
}

template<size_t ndim>
size_t  
Heuristic_classifier<ndim>::num_ints_needed(const size_t num_regions){//move to pagani utils, has nothing to do with classifying
    const size_t scanned = num_regions;
    const size_t subDivDim = 2 * num_regions;
    const size_t activeBisectDim = num_regions;
    return activeBisectDim + subDivDim + scanned;
}

template<size_t ndim>
size_t  
Heuristic_classifier<ndim>::num_doubles_needed(const size_t num_regions){//move to pagani utils, has nothing to do with classifying
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

template<size_t ndim>
void  
Heuristic_classifier<ndim>::store_estimate(const double estimate){
    estimates_from_last_iters[0] = estimates_from_last_iters[1];
    estimates_from_last_iters[1] = estimates_from_last_iters[2];
    estimates_from_last_iters[2] = estimate;
    iters_collected++;
}

template<size_t ndim>
bool  
Heuristic_classifier<ndim>::estimate_converged(){
    //the -1 is because iters_collected++ is 1 at iteration 0 and I don't want to start counting at -1
    if(iters_collected - 1 < min_iters_for_convergence || !sigDigitsSame())
        return false;
    return true;
}

template<size_t ndim>
bool
Heuristic_classifier<ndim>::sigDigitsSame()
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
          
    for (int i = 0; i < required_digits + 1 && sigDigits < required_digits &&  verdict == true;  ++i) {
        verdict =
              current[i] == last[i] && last[i] == second_to_last[i] ? true : false;

            sigDigits += (verdict == true && current[i] != '.') ? 1 : 0;
    }
    return verdict;
} 

template<size_t ndim>
void 
Heuristic_classifier<ndim>::apply_threshold(sycl::queue& q, Classification_res& res, const double* errorests, const size_t num_regions){
            
    auto int_division = [](int x, int y){
        return static_cast<double>(x)/static_cast<double>(y);
    };
            
    set_true_for_larger_than<double>(q, errorests, res.threshold, num_regions, res.active_flags);
    //make the lines below another function compute num active regions
    res.num_active  = static_cast<size_t>(quad::reduction<double>(q, res.active_flags, num_regions));  
    //make the lines below to another function, mem_reqs_pass_()
    res.percent_mem_active = int_division(res.num_active, num_regions);
    res.percent_mem_active = int_division(res.num_active, num_regions);
	//std::cout<<"res.num_active:"<< res.num_active << std::endl;
	//std::cout<<"res.percent_mem_active:"<<res.percent_mem_active<<std::endl;
	//std::cout<<"max_active_regions_percentage:"<<max_active_regions_percentage<<std::endl;
	res.pass_mem = res.percent_mem_active <= max_active_regions_percentage;
}

template<size_t ndim>
void 
Heuristic_classifier<ndim>::evaluate_error_budget(sycl::queue& q, Classification_res& res, 
    double* error_estimates,
    double* active_flags, 
    const size_t num_regions,
    const double target_error, 
    const double active_errorest, 
    const double iter_finished_errorest, 
    const double total_f_errorest,
    const double max_percent_err_budget){
            
    const double extra_f_errorest = active_errorest - quad::dot_product<double>(q, error_estimates, active_flags, num_regions) - iter_finished_errorest;  
    const double error_budget = target_error - total_f_errorest;
    res.pass_errorest_budget = extra_f_errorest <= max_percent_err_budget * error_budget;
    res.finished_errorest =  extra_f_errorest;  
}

template<size_t ndim>
void
Heuristic_classifier<ndim>::get_larger_threshold_results(sycl::queue& q, Classification_res& thres_search, 
    const double* active_flags, //this paramter is not needed
    const double* errorests, 
    const size_t num_regions){
                
    thres_search.pass_mem = false;
    const size_t max_attempts = 20;
    size_t counter = 0;
            
    while(!thres_search.pass_mem && counter < max_attempts){
        apply_threshold(q, thres_search, errorests, num_regions);
        if(!thres_search.pass_mem){
            
            thres_search.threshold_range.low = thres_search.threshold;
            thres_search.increase_threshold();   
        }
        counter++;
    }
}

template<size_t ndim>
bool 
Heuristic_classifier<ndim>::classification_criteria_met(sycl::queue& q, const size_t num_regions){
    double ratio = static_cast<double>(device_mem_required_for_full_split(num_regions))/static_cast<double>(free_device_mem(num_regions, ndim));
          
    if(ratio > 1.)
        return true;
    else if(ratio > .1 && estimate_converged())
        return true;
    else
        return false;
}


template<size_t ndim>
Classification_res 
Heuristic_classifier<ndim>::classify(sycl::queue& q, 
            double* active_flags,
            double* errorests,
            const size_t num_regions,
            const double iter_errorest, 
            const double iter_finished_errorest,
            const double total_finished_errorest){
                           
            Classification_res thres_search = (device_array_min_max<double>(q, errorests, num_regions));
            thres_search.data_allocated = true;
            
            const double min_errorest = thres_search.threshold_range.low;
            const double max_errorest = thres_search.threshold_range.high;
            thres_search.threshold = iter_errorest/num_regions;
            thres_search.active_flags = sycl::malloc_device<double>(num_regions, q);
            const double target_error = abs(estimates_from_last_iters[2]) * epsrel;

            const size_t max_num_thresholds_attempts = 20;
            size_t num_thres_increases = 0;
            size_t num_thres_decreases = 0;
            size_t max_thres_increases = 20;
            
            int threshold_changed = 0; //keeps track of where the threshold is being pulled (left or right)
            do{
				std::cout<<"classifying"<<std::endl;
                if(!thres_search.pass_mem && num_thres_increases <= max_thres_increases){
                    
                    get_larger_threshold_results(q, thres_search, active_flags, errorests, num_regions);
                    num_thres_increases++;
                }
                //put the next in an else statement
                
                if(thres_search.pass_mem){
                    evaluate_error_budget(q, thres_search, 
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
                    max_active_regions_percentage += .1;
                }
                else if(exhausted_attempts){   
                    break;
                }
                
                
            }while(!thres_search.pass_mem ||  !thres_search.pass_errorest_budget);
            
            if(!thres_search.pass_mem || !thres_search.pass_errorest_budget){
                sycl::free(thres_search.active_flags, q);
            }
            
            thres_search.max_budget_perc_to_cover = max_percent_error_budget;
            thres_search.max_active_perc = max_active_regions_percentage;
            
            max_active_regions_percentage = .5;
            max_percent_error_budget = .25;
            
            return thres_search;
}

#endif
