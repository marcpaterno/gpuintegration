#ifndef KOKKOSCUHRE_PHASES_CUH
#define KOKKOSCUHRE_PHASES_CUH

#include "quad.h"
#include "Sample.cuh"    

template<int NDIM>
__device__
void
ActualCompute(ViewVectorDouble generators, double* g, const Structures<double>& constMem, size_t feval_index, size_t total_feval, const member_type team_member){
    
	for (int dim = 0; dim < NDIM; ++dim) {
		g[dim] = 0;
    }
	
	int threadIdx = team_member.team_rank();
    int blockIdx = team_member.league_rank();  
	
    int posCnt = constMem._gpuGenPermVarStart(feval_index + 1) -
                 constMem._gpuGenPermVarStart(feval_index);
    int gIndex = constMem._gpuGenPermGIndex(feval_index);   
   
    for (int posIter = 0; posIter < posCnt; ++posIter) {
        int pos = constMem._gpuGenPos((constMem._gpuGenPermVarStart(feval_index)) + posIter);
        int absPos = abs(pos);
          
        if (pos == absPos) { 
			g[absPos - 1] = constMem._gpuG(gIndex * NDIM + posIter);
        } else {
            g[absPos - 1] = -constMem._gpuG(gIndex * NDIM + posIter);
        }
    }
        
    for(int dim=0; dim<NDIM; dim++){
        generators(total_feval*dim + feval_index) = g[dim];
    }        
}

template<int NDIM>
void
ComputeGenerators(ViewVectorDouble generators, size_t FEVAL, const Structures<double> constMem){
	uint32_t nBlocks = 1;
    uint32_t nThreads = BLOCK_SIZE;
	
	Kokkos::parallel_for( "Phase1", team_policy(nBlocks, nThreads), KOKKOS_LAMBDA (const member_type team_member) {
            
        int threadIdx = team_member.team_rank();
        int blockIdx = team_member.league_rank();        
        
		size_t perm = 0;
		double g[NDIM];
		
		for (size_t dim = 0; dim < NDIM; ++dim) {
			g[dim] = 0;
		}
    
		size_t feval_index = perm * BLOCK_SIZE + threadIdx;
		if (feval_index < FEVAL) {
			ActualCompute<NDIM>(generators, g, constMem, feval_index, FEVAL, team_member);
		}
		
		team_member.team_barrier();
		
		for (perm = 1; perm < FEVAL / BLOCK_SIZE; ++perm) {
			int feval_index = perm * BLOCK_SIZE + threadIdx;
			ActualCompute<NDIM>(generators, g, constMem, feval_index, FEVAL, team_member);
		}
		
		team_member.team_barrier();
		
		feval_index = perm * BLOCK_SIZE + threadIdx;
		if (feval_index < FEVAL) {
			int feval_index = perm * BLOCK_SIZE + threadIdx;
			ActualCompute<NDIM>(generators, g, constMem, feval_index, FEVAL, team_member);
		}
		
		team_member.team_barrier();          
    });
}

template <typename IntegT, int NDIM>
__device__ void
INIT_REGION_POOL(IntegT d_integrand, 
                 ViewVectorDouble dRegions, 
                 ViewVectorDouble dRegionsLength,
                 size_t numRegions,
                 const Structures<double>& constMem,
                 int FEVAL,
                 int NSETS,                
                 ViewVectorDouble lows,
                 ViewVectorDouble highs,
                 int iteration,
                 int depth,
                 constViewVectorDouble generators,        
				 Kokkos::View<Region<NDIM>*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > sRegionPool,
                 const member_type team_member){
    
    typedef Kokkos::View<Region<NDIM>*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewRegion;
    
    ScratchViewDouble vol(team_member.team_scratch(0), 1);
    ScratchViewDouble Jacobian(team_member.team_scratch(0), 1);
    ScratchViewDouble ranges(team_member.team_scratch(0), NDIM);
    ScratchViewInt maxDim(team_member.team_scratch(0), 1);
    ScratchViewGlobalBounds sBound(team_member.team_scratch(0), NDIM);
	
    //int threadIdx = team_member.team_rank();
		   
    if(team_member.team_rank() == 0){
		
		int blockIdx = team_member.league_rank();     
        Jacobian(0) = 1;
        double maxRange = 0;
        for (int dim = 0; dim < NDIM; ++dim) {
            double lower = dRegions(dim * numRegions + blockIdx);
            sRegionPool(0).bounds[dim].lower = lower;
            sRegionPool(0).bounds[dim].upper =  lower + dRegionsLength(dim * numRegions + blockIdx);
                    
            sBound(dim).unScaledLower = lows(dim);
            sBound(dim).unScaledUpper = highs(dim);
            ranges(dim) = sBound(dim).unScaledUpper - sBound(dim).unScaledLower;
            sRegionPool(0).div = depth; 
            
            double range = sRegionPool(0).bounds[dim].upper - lower;
            Jacobian(0) = Jacobian(0) * ranges(dim);		
            
            if(range > maxRange){
                maxDim(0) = dim;
                maxRange = range;        
            }
        }
        vol(0) = ldexp(1., -depth);
    }

    int sIndex = 0;
    team_member.team_barrier();                                  
    Sample<IntegT, NDIM>(d_integrand, 
					     sIndex,  
						 constMem,
						 FEVAL,
						 NSETS,
						 sRegionPool,
						 vol, 
						 maxDim, 
						 ranges, 
						 Jacobian, 
						 generators,
						 //sdata,
						 sBound,
						 team_member);
    team_member.team_barrier();
}

template<typename IntegT, int NDIM>
void
INTEGRATE_GPU_PHASE1(IntegT d_integrand, 
					 ViewVectorDouble dRegions, 
					 ViewVectorDouble dRegionsLength, 
					 size_t numRegions, 
					 ViewVectorDouble dRegionsIntegral, 
					 ViewVectorDouble dRegionsError,
					 ViewVectorDouble activeRegions,
					 ViewVectorInt subDividingDimension,
					 double epsrel,
					 double epsabs,
					 Structures<double> constMem,
					 int FEVAL,
					 int NSETS,
					 ViewVectorDouble lows,
					 ViewVectorDouble highs,
					 int iteration,
					 int depth,
					 constViewVectorDouble generators){
        
        uint32_t nBlocks = numRegions;
        uint32_t nThreads = BLOCK_SIZE;
        typedef Kokkos::View<Region<NDIM>*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewRegion;
        
        int shMemBytes = ScratchViewInt::shmem_size(1) +   //for maxDim
                         ScratchViewDouble::shmem_size(1) +   //for vol
                         ScratchViewDouble::shmem_size(1) +   //for Jacobian
                         ScratchViewDouble::shmem_size(NDIM) +  //for ranges
                         ScratchViewRegion::shmem_size(1)+ //how come shmem_size doesn't return size_t? the tutorial exercise was returning an int too
                         ScratchViewGlobalBounds::shmem_size(NDIM)+ //for sBound
                         ScratchViewDouble::shmem_size(BLOCK_SIZE); //for sdata
                
                         
                         
		Kokkos::parallel_for( "Phase1", team_policy(nBlocks, nThreads).set_scratch_size(0, Kokkos::PerTeam(shMemBytes)), KOKKOS_LAMBDA (const member_type team_member) {
            
			ScratchViewRegion sRegionPool(team_member.team_scratch(0), 1);
				
            INIT_REGION_POOL<IntegT, NDIM>(d_integrand,
				dRegions, dRegionsLength, numRegions, constMem, FEVAL, NSETS, 
				lows, highs, iteration, depth, generators, sRegionPool, team_member);
			team_member.team_barrier();
			
			if (team_member.team_rank() == 0) {           
                activeRegions(team_member.league_rank()) = 1.; 
                subDividingDimension(team_member.league_rank()) = sRegionPool(0).result.bisectdim;
                dRegionsIntegral(team_member.league_rank()) = sRegionPool(0).result.avg;;
                dRegionsError(team_member.league_rank()) =  sRegionPool(0).result.err;;
			}
        });     
}
  
#endif
