#include "externals/catch2/catch.hpp"
#include "kokkos/pagani/quad/GPUquad/Region_characteristics.cuh"
#include "kokkos/pagani/quad/GPUquad/Region_estimates.cuh"
#include "kokkos/pagani/quad/GPUquad/Sub_region_filter.cuh"

TEST_CASE("All finished regions")
{
  constexpr bool use_custom = true;	
  const size_t num_regions = 10;
  Region_characteristics<2> regs(num_regions);  
  Sub_regions_filter<double, 2, use_custom> filter(num_regions);
  auto active_regions = Kokkos::create_mirror_view(regs.active_regions);
  
  for(int i=0; i < num_regions; ++i)
      active_regions[i] = 0.;  

  Kokkos::deep_copy(regs.active_regions, active_regions);
  size_t num_active = filter.get_num_active_regions(regs.active_regions, num_regions);    
  CHECK(num_active == 0);  
}

TEST_CASE("No finished regions")
{
  constexpr bool use_custom = true;	
  const size_t num_regions = 10;
  Region_characteristics<2> regs(num_regions);  
  Sub_regions_filter<double, 2, use_custom> filter(num_regions);
  auto active_regions = Kokkos::create_mirror_view(regs.active_regions);
  for(int i=0; i < num_regions; ++i)
      active_regions[i] = 1.;  
  
  Kokkos::deep_copy(regs.active_regions, active_regions);
  size_t num_active = filter.get_num_active_regions(regs.active_regions, num_regions);    
  CHECK(num_active == 10);
}


TEST_CASE("Some sub-regions are finished")
{
  constexpr bool use_custom = true;	
  const size_t num_ranges_with_all_finished = 5;
  std::array<std::pair<size_t, size_t>, num_ranges_with_all_finished> finished_ranges = {{{0, 50},{63, 71},{ 101, 121},{124, 125},{127, 129}}};
 
  const size_t num_regions = 1000;
  Region_characteristics<2> regs(num_regions);  
  Sub_regions_filter<double, 2, use_custom> filter(num_regions);
  
  auto active_regions = Kokkos::create_mirror_view(regs.active_regions);
  for(int i=0; i < num_regions; ++i)
      active_regions[i] = 1.;  
  
  for(auto range : finished_ranges){
      //set finished regions from ranges
      for(int i=range.first; i <= range.second; ++i)
          active_regions[i] = 0.;       
  }  
    
    
  size_t num_true_active = 1000;
  for(int i = 0; i < num_ranges_with_all_finished; ++i){
      num_true_active -= finished_ranges[i].second - finished_ranges[i].first + 1;
  }  
  
  Kokkos::deep_copy(regs.active_regions, active_regions);
  size_t num_active = filter.get_num_active_regions(regs.active_regions, num_regions);    
  CHECK(num_active == num_true_active);
}

TEST_CASE("Only first and last are finished")
{ 
  constexpr bool use_custom = true;
  const size_t num_regions = 1000;
  Region_characteristics<2> regs(num_regions);  
  Sub_regions_filter<double, 2, use_custom> filter(num_regions);
  
  auto active_regions = Kokkos::create_mirror_view(regs.active_regions);
  for(int i=0; i < num_regions; ++i)
      active_regions[i] = 1.;  
    
  active_regions[0] = 0.;  
  active_regions[num_regions-1] = 0.;  
  Kokkos::deep_copy(regs.active_regions, active_regions);
  size_t num_active = filter.get_num_active_regions(regs.active_regions, num_regions);          
  CHECK(num_active == 998);
}
