#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"
#include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "kokkos/pagani/quad/GPUquad/PaganiUtils.cuh"
#include "common/integration_result.hh"
#include <vector>
#include <array>



TEST_CASE("Compute finished estimates")
{
    constexpr size_t ndim = 2;
    size_t num_regions = 100;
    Region_estimates<double, ndim> estimates(num_regions);
    Region_characteristics<ndim> characteristics(num_regions);
    
    double uniform_estimate = 3.2;
    double uniform_errorest =  .00001;
    
    size_t nregions = estimates.size;
    for(size_t i = 0; i < nregions; ++i){
        estimates.integral_estimates[i] = uniform_estimate;
        estimates.error_estimates[i] = uniform_errorest;
    }
    
    SECTION("All finished regions")
    {
        for(size_t i = 0; i < nregions; ++i){
            characteristics.active_regions[i] = 0.;
        }
        
        numint::integration_result true_iter_estimate;
        true_iter_estimate.estimate = uniform_estimate * static_cast<double>(nregions);
        true_iter_estimate.errorest = uniform_errorest * static_cast<double>(nregions);
        
        numint::integration_result test = compute_finished_estimates(estimates, characteristics, true_iter_estimate);
        CHECK(true_iter_estimate.estimate == Approx(test.estimate));
        CHECK(true_iter_estimate.errorest == Approx(test.errorest));
    }
    
    SECTION("Few active regions bundled together")
    {
        numint::integration_result true_iter_estimate;
        true_iter_estimate.estimate = uniform_estimate * static_cast<double>(nregions);
        true_iter_estimate.errorest = uniform_errorest * static_cast<double>(nregions);
        
        size_t first_index = 11;    //first active region
        size_t last_index = 17;     //last active region
        double num_true_active_regions = static_cast<double>(last_index - first_index + 1);
        
        numint::integration_result true_iter_finished_estimate;
        true_iter_finished_estimate.estimate = uniform_estimate * static_cast<double>(nregions) - uniform_estimate * num_true_active_regions;
        true_iter_finished_estimate.errorest = uniform_errorest * static_cast<double>(nregions) - uniform_errorest * num_true_active_regions;
        
        for(size_t i = 0; i < nregions; ++i){
            bool in_active_range = static_cast<bool>(i >= first_index && i <= last_index);
            characteristics.active_regions[i] =  in_active_range ? 1. : 0.;
        }
        
        numint::integration_result test = compute_finished_estimates(estimates, characteristics, true_iter_estimate);
        CHECK(test.estimate == Approx(true_iter_finished_estimate.estimate));
        CHECK(test.errorest == Approx(true_iter_finished_estimate.errorest));
    }
}
