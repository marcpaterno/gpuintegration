#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include <iostream>
#include "oneAPI/pagani/quad/GPUquad/PaganiUtils.dp.hpp"
#include "oneAPI/pagani/quad/util/cudaMemoryUtil.h"

TEST_CASE("Positive Constant Function")
{
    class Positive{
      public:
        Positive() = default;
    
        double
        operator()(double x, double y){
            return 5.5;
        }
    };
    
    size_t partitions_per_axis = 2;
    constexpr size_t ndim = 2;
    Positive integrand;
    double true_answer = 5.5;
    Cubature_rules<ndim> cubature_rules;
    Sub_regions<ndim> sub_regions(partitions_per_axis);
    
    
    Region_estimates<ndim> subregion_estimates(sub_regions.size);
    Region_characteristics<ndim> region_characteristics(sub_regions.size);
        
    Positive* d_integrand = make_gpu_integrand<Positive>(integrand);
    int it = 0;    
    cuhreResult<double> res = cubature_rules.apply_cubature_integration_rules<Positive>(d_integrand, it, &sub_regions, &subregion_estimates, &region_characteristics);
    CHECK(res.estimate == Approx(true_answer));  
    CHECK(array_values_smaller_than_val<int, size_t>(region_characteristics.sub_dividing_dim, sub_regions.size, ndim));
}


TEST_CASE("Negative Constant Function")
{
    class Negative{
      public:
        Negative() = default;
    
        double
        operator()(double x, double y){
            return -5.5;
        }
    };
    
    size_t partitions_per_axis = 2;
    constexpr size_t ndim = 2;
    Negative integrand;
    double true_answer = -5.5;
    Cubature_rules<ndim> cubature_rules;
    Sub_regions<ndim> sub_regions(partitions_per_axis);
    
    
    Region_estimates<ndim> subregion_estimates(sub_regions.size);
    Region_characteristics<ndim> region_characteristics(sub_regions.size);
        
    Negative* d_integrand = make_gpu_integrand<Negative>(integrand);
    int it = 0;       
    cuhreResult<double> res = cubature_rules.apply_cubature_integration_rules<Negative>(d_integrand, it, &sub_regions, &subregion_estimates, &region_characteristics);
    CHECK(res.estimate == Approx(true_answer));  
    CHECK(array_values_smaller_than_val<int, size_t>(region_characteristics.sub_dividing_dim, sub_regions.size, ndim));
}