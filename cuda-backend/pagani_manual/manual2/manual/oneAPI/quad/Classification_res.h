#ifndef ONE_API_CLASSIFICATION_RES
#define ONE_API_CLASSIFICATION_RES

#include "oneAPI/quad/util/MemoryUtil.h"

class Classification_res{
    public:
    Classification_res() = default;
    Classification_res(quad::Range<double> some_range):threshold_range(some_range){}
    
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
    quad::Range<double> threshold_range; //change to threshold_range
    double* active_flags = nullptr;
    size_t num_active = 0;
    double finished_errorest = 0.;
    
    double max_budget_perc_to_cover = .25;
    double max_active_perc = .5;
    bool data_allocated = false;
};

#endif
