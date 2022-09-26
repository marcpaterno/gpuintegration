//#include <CL/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include "oneAPI/quad/Cubature_rules.h"
#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/GPUquad/Rule.h"
#include "oneAPI/quad/GPUquad/Phases.h"
#include "oneAPI/quad/Cubature_rules.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>
#include "oneAPI/quad/Workspace.h"

#include "oneAPI/demos/demo_utils.h"

#include "oneapi/mkl.hpp"
#include "oneapi/tbb.h"
#include <limits>

void ShowDevice(queue &q) {
  // Output platform and device information.
  auto device = q.get_device();
  auto p_name = device.get_platform().get_info<info::platform::name>();
  std::cout << std::setw(20) << "Platform Name: " << p_name << "\n";
  auto p_version = device.get_platform().get_info<info::platform::version>();
  std::cout << std::setw(20) << "Platform Version: " << p_version << "\n";
  auto d_name = device.get_info<info::device::name>();
  std::cout << std::setw(20) << "Device Name: " << d_name << "\n";
  auto max_work_group = device.get_info<info::device::max_work_group_size>();
  std::cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
  auto max_compute_units = device.get_info<info::device::max_compute_units>();
  std::cout << std::setw(20) << "Max Compute Units: " << max_compute_units << "\n\n";


  std::cout << "wgroup_size " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;

  std::cout << "local_mem_size " <<  device.get_info<sycl::info::device::local_mem_size>() << std::endl;

  std::cout << "max_compute_units  " <<   device.get_info<sycl::info::device::max_compute_units>() << std::endl;

  std::cout << "max_mem_alloc_size " << device.get_info<sycl::info::device::max_mem_alloc_size>() << std::endl;

}

//using namespace sycl;

static constexpr size_t N = 100000; // global size
static constexpr size_t B = 64; // work-group size

void warmup(sycl::queue &q){
    size_t num_elems = 10;
    double* vals = sycl::malloc_shared<double>(num_elems, q);
    printf("in warpup\n");
    q.submit([&](auto &cgh) {
        cgh.parallel_for(sycl::range<1>(10), [=](sycl::id<1> tid){
            //vals[tid] = 1.;
        });
     }).wait();
    
    sycl::free(vals, q);
    printf("end of warmpup\n");
}


int main() {
  queue q;
  std::cout << "Devicee : " << q.get_device().get_info<info::device::name>() << "\n";
  ShowDevice(q);  

  Workspace<8> workspace(q);
        
  
  constexpr size_t ndim = 3;
  double divs_per_dim = 2.;
  const int nregions = pow(divs_per_dim, ndim);
  const double length = 1./divs_per_dim;
    
  double* LeftCoord = sycl::malloc_shared<double>(nregions * ndim, q);
  double* Length = sycl::malloc_shared<double>(nregions * ndim, q);
    
  std::cout<<"nregions:"<<nregions<<std::endl;
    
  for(int i=0; i < ndim*nregions; ++i){
    LeftCoord[i] = 0.;
    Length[i] = 0.;
  }
    
  q.submit([&](handler &h) {
    
        
	     h.parallel_for(sycl::range<1>(nregions),
			    [=](sycl::id<1> _reg){
			      size_t reg = _reg[0];
		
			      for(int dim = 0; dim < ndim; ++dim){
				size_t _id = (size_t)(static_cast<int>(reg) /  (int)pow(divs_per_dim, (double)dim)) % (int)divs_per_dim;
				LeftCoord[nregions * dim + static_cast<int>(reg)] = static_cast<double>(_id) * length;  
				Length[nregions * dim + static_cast<int>(reg)] = length;

			      }
                 
                 
			    });

	     //printf("after parallel_for\n");
    
	   }).wait();        
    
    
    
    
  /*for(int i =0; i <  nregions; ++i){
    
    printf(" region %i (%f,%f) (%f, %f) \n", 
	   i, LeftCoord[i+0*nregions], LeftCoord[i+0*nregions] + Length[i+0*nregions], 
	   LeftCoord[i+1*nregions], LeftCoord[i+1*nregions] + Length[i+1*nregions]); 
    
  }*/
    
  size_t size = nregions;  
  return 0;
}
