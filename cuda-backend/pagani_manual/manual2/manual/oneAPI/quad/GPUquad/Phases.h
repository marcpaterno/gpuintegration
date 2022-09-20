#ifndef ONE_API_QUAD_GPUQUAD_PHASES_CUH
#define ONE_API_QUAD_GPUQUAD_PHASES_CUH

#include "oneAPI/quad/util/cudaArray.h"
#include "oneAPI/quad/util/cudaApply.h"
#include "oneAPI/quad/GPUquad/Sample.h"
#include "oneAPI/quad/util/MemoryUtil.h"
#include <CL/sycl.hpp>
#include "oneAPI/quad/Rule_Params.h"

#include <stdarg.h>
#include <stdio.h>
#include <tuple>


namespace quad {
    //using shared = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>;  
  
    template <typename IntegT, int ndim, int blockDim/*, int warp_size*/>
    void
    init_region_pool(IntegT* d_integrand,
                   double* dRegions,
                   double* dRegionsLength,
                   size_t num_regions,
                   const Structures<double> constMem,
                   Region<ndim> sRegionPool[],
                   GlobalBounds sBound[],
                   double* lows,
                   double* highs,
				   double* generators,
				   sycl::nd_item<1> item,
				   double* scratch,
				   double* sdata,
				   double* jacobian,
                   int* max_dim,
                   double*vol,
                   double* ranges){
     
     size_t index = item.get_group(0);    
    if (item.get_local_id(0) == 0) {
      vol[0] = 1.;
      jacobian[0] = 1.;
      double maxRange = 0;
      for (int dim = 0; dim < ndim; ++dim) {
        double lower = dRegions[dim * num_regions + index];
        sRegionPool[0].bounds[dim].lower = lower;
        sRegionPool[0].bounds[dim].upper = lower + dRegionsLength[dim * num_regions + index];

        sBound[dim].unScaledLower = lows[dim];
        sBound[dim].unScaledUpper = highs[dim];
        ranges[dim] = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;
        double range = sRegionPool[0].bounds[dim].upper - lower;
        vol[0] *= range;
        jacobian[0] = jacobian[0] * ranges[dim];
          
        if (range > maxRange) {
          max_dim[0] = dim;
          maxRange = range;
        }
      }
    }
	
	item.barrier(/*sycl::access::fence_space::local_space*/);  
    sample_region_block<IntegT, ndim, blockDim>(d_integrand,
                    constMem,
                    //feval,
                    //nsets,
                    sRegionPool,
                    sBound,
                    vol,
                    max_dim,
                    ranges,
                    jacobian,
					generators,
					item,
					scratch,
                    sdata );
    item.barrier(sycl::access::fence_space::local_space);
  }
    
  template <typename F, int ndim, int blockDim/*, int warp_size*/>
  void
  integrate_kernel(queue& q,
                F* d_integrand,
                double* dRegions,
                double* dRegionsLength,
                size_t num_regions,
                double* dRegionsIntegral,
                double* dRegionsError,
                double* activeRegions,
                int* subDividingDimension,
                double epsrel,
                double epsabs,
                const Structures<double> constMem,
                double* lows,
                double* highs,
				double* generators)
  {
		//F* integrand = malloc_shared<F>(1, q);
		sycl::event e = q.submit([&](sycl::handler& cgh) {
		
        shared<int> max_dim(sycl::range(1), cgh);
        shared<double> jacobian(sycl::range(1), cgh);
        shared<double> vol(sycl::range(1), cgh);
        shared<double> ranges(sycl::range(ndim), cgh);
        shared<GlobalBounds> sBound(sycl::range(ndim), cgh);
        shared<Region<ndim>> sRegionPool(sycl::range(1), cgh);
        shared<double> sdata(sycl::range(blockDim), cgh);
        
        shared<double> shared(sycl::range(8), cgh); 
		
        cgh.parallel_for(sycl::nd_range<1>(num_regions * blockDim, blockDim), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]]{
            
            init_region_pool<F, ndim, blockDim/*, warp_size*/>(d_integrand,
                   dRegions,
                   dRegionsLength,
                   num_regions,
                   constMem,
                   sRegionPool.get_pointer(),
                   sBound.get_pointer(),
                   lows,
                   highs,
				   generators,
				   item,
				   shared.get_pointer(),
				   sdata.get_pointer(),
				   jacobian.get_pointer(),
				   max_dim.get_pointer(),
                   vol.get_pointer(),
                   ranges.get_pointer());
            
                               
            if(item.get_local_id() == 0){  
			  activeRegions[item.get_group(0)] = 1.;
              subDividingDimension[item.get_group(0)] = sRegionPool[0].result.bisectdim;
              dRegionsIntegral[item.get_group(0)] = sRegionPool[0].result.avg;
              dRegionsError[item.get_group(0)] = sRegionPool[0].result.err;
            }
        });
        
        
    });    
		
	q.wait();
	double time = (e.template get_profiling_info<sycl::info::event_profiling::command_end>()  -   
	e.template get_profiling_info<sycl::info::event_profiling::command_start>());
	std::cout<< "time:" << std::scientific << time/1.e6 << "," << ndim << ","<< num_regions << std::endl;
  }          
}

#endif
