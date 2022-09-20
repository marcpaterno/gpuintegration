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
  
    template <int ndim, int blockDim, int warp_size>
    void
    init_region_pool(sycl::nd_item<1> item,
                   double* dRegions,
                   double* dRegionsLength,
                   size_t num_regions,
                   const Rule_Params<ndim>& constMem,
                   int feval,
                   int nsets,
                   shared<Region<ndim>> sRegionPool,
                   shared<GlobalBounds> sBound,
                   double* lows,
                   double* highs,
                   shared<double> vol,
                   shared<int> max_dim,
                   shared<double> ranges,
                   shared<double> jacobian){
        
    size_t work_group_tid = item.get_local_id();
    const size_t work_group_id = item.get_group_linear_id();
        
    if (work_group_tid == 0) {
      vol[0] = 1.;
      jacobian[0] = 1.;
      double maxRange = 0;
      for (int dim = 0; dim < ndim; ++dim) {
        double lower = dRegions[dim * num_regions + work_group_id];
        sRegionPool[0].bounds[dim].lower = lower;
        sRegionPool[0].bounds[dim].upper = lower + dRegionsLength[dim * num_regions + work_group_id];

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
        
        
      //for(int i=0; i < blockDim/32; ++i)
      //    scratch[i] = 0.;
    }
  }
    
  template <typename F, int ndim, int blockDim, int warp_size>
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
                const Rule_Params<ndim>& constMem,
                double* lows,
                double* highs)
  {
    F* integrand = malloc_shared<F>(1, q);
	sycl::event e = q.submit([&](sycl::handler& cgh) {

        //sycl::stream str(0, 0, cgh);
        shared<int> max_dim(sycl::range(1), cgh);
        shared<double> jacobian(sycl::range(1), cgh);
        shared<double> vol(sycl::range(1), cgh);
        shared<double> ranges(sycl::range(ndim), cgh);
        shared<GlobalBounds> sBound(sycl::range(ndim), cgh);
        shared<Region<ndim>> sRegionPool(sycl::range(1), cgh);
        shared<double> sdata(sycl::range(blockDim), cgh);
        
        size_t work_group_size = blockDim;
        const size_t num_sub_groups = work_group_size/warp_size;
        size_t total_threads = num_regions * work_group_size; 
        size_t feval = get_feval<ndim>();
        constexpr int nsets = 9;
        
        shared<double> scratch(sycl::range(num_sub_groups), cgh); 
        cgh.parallel_for(sycl::nd_range<1>(total_threads, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]]{
            
            init_region_pool<ndim, blockDim, warp_size>(item,
                   dRegions,
                   dRegionsLength,
                   num_regions,
                   constMem,
                   feval,
                   nsets,
                   sRegionPool,
                   sBound,
                   lows,
                   highs,
                   vol,
                   max_dim,
                   ranges,
                   jacobian//,
                   //scratch,
                   /*str*/);
            
            item.barrier(sycl::access::fence_space::local_space);  
            int sIndex = 0;
            sample_region_block<F, ndim, blockDim>( item,
                    integrand,
                    sIndex,
                    constMem,
                    feval,
                    nsets,
                    sRegionPool,
                    sBound,
                    vol,
                    max_dim,
                    ranges,
                    jacobian,
                    sdata,
                    scratch);
            item.barrier(sycl::access::fence_space::local_space);  
            
            const size_t work_group_id = item.get_group_linear_id();
            const size_t work_group_tid = item.get_local_id();
                    
            if(work_group_tid == 0){  
			  activeRegions[work_group_id] = 1.;
              subDividingDimension[work_group_id] = sRegionPool[0].result.bisectdim;
              dRegionsIntegral[work_group_id] = sRegionPool[0].result.avg;
              dRegionsError[work_group_id] = sRegionPool[0].result.err;
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
