#include "../util/cudaApply.cuh"
#include "../util/cudaArray.cuh"

namespace quad{
  template<typename T>
    __device__ 
    T Sq(T x) {
    return x*x;
  }


  __device__
double warpReduceSum(double val) {
	val += __shfl_down_sync(0xffffffff, val, 16, 32);
	val += __shfl_down_sync(0xffffffff, val, 8, 32);
	val += __shfl_down_sync(0xffffffff, val, 4, 32);
	val += __shfl_down_sync(0xffffffff, val, 2, 32);
	val += __shfl_down_sync(0xffffffff, val, 1, 32);
	return val;
}


  template<typename T>
    __device__
    T
    computeReduce(T sum){
		sdata[threadIdx.x] = sum;
		__syncthreads();
  
		// contiguous range pattern
		for(size_t offset = blockDim.x / 2; offset > 0; offset >>= 1){
			if(threadIdx.x < offset){
				sdata[threadIdx.x] += sdata[threadIdx.x + offset];
			}
			__syncthreads();
		}
    
		return sdata[0];
	}

    template <typename IntegT, typename T, int NDIM>
    __device__
    void
    computePermutation(IntegT* d_integrand, int pIndex, Bounds *b, T *g,gpu::cudaArray<T, NDIM>& x, T *sum, Structures<T> constMem){
		
		for(int dim = 0; dim < NDIM; ++dim){
		  g[dim]=0;
		}
		
		int posCnt = __ldg(&constMem.gpuGenPermVarStart[pIndex+1])-__ldg(&constMem.gpuGenPermVarStart[pIndex]);
		int gIndex = __ldg(&constMem.gpuGenPermGIndex[pIndex]);
		
		T *lG = &constMem.gpuG[gIndex*NDIM];
		for(int posIter = 0; posIter < posCnt; ++posIter){
		  int pos = __ldg(&constMem.gpuGenPos[__ldg(&constMem.gpuGenPermVarStart[pIndex])+posIter]);
		  int absPos = abs(pos);
		  if(pos == absPos)
			g[absPos-1] =  lG[posIter];
		  else
			g[absPos-1] =  -lG[posIter];
		}

		 // if(threadIdx.x == 0)
		 //   printf("This is block:\n");
		
		//sBound is shared memory
		T jacobian = 1;
		for(int dim = 0; dim < NDIM; ++dim ){
		  x[dim] = (.5 + g[dim])*b[dim].lower + (.5 - g[dim])*b[dim].upper;
		  T range = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;
		  jacobian = jacobian*range;
		  x[dim] =  sBound[dim].unScaledLower + x[dim]*range;
		}
		
        //if(blockIdx.x == 10 && threadIdx.x < 10)
        //    printf("%i, %i x:%f, %f, %f, %f, %f, %f, %f\n", blockIdx.x, threadIdx.x, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
		T fun = gpu::apply(*d_integrand, x);
		fun = fun*jacobian;
		sdata[threadIdx.x] = fun;
		//printf("%d %lf\n",pIndex, fun);

		T *weight = &constMem.cRuleWt[gIndex*NRULES];
	  
		for(int rul = 0; rul < NRULES; ++rul ){
		  sum[rul] += fun*weight[rul];
		}
  }

  //BLOCK SIZE has to be atleast 4*DIM+1 for the first IF 
    template <typename IntegT, typename T, int NDIM>
    __device__
    void
    SampleRegionBlock(IntegT* d_integrand, int sIndex, Structures<T> constMem, int FEVAL,  int NSETS,  Region<NDIM> sRegionPool[]){ 
    
	typedef Region<NDIM> Region;
	//read
    Region *const region = (Region *)&sRegionPool[sIndex];
	
    T vol = ldexp(1., -region->div); // this means: 1*2^(-region->div)
    //if(threadIdx.x == 0 && blockIdx.x == 0)
    //    printf("vol:%.15e with div:%i\n", vol, region->div);
    T g[NDIM];
    gpu::cudaArray<double, NDIM> x;
    int perm = 0;
	
    T ratio = Sq(__ldg(&constMem.gpuG[2*NDIM])/__ldg(&constMem.gpuG[1*NDIM]));
    int offset = 2*NDIM;
    int maxdim = 0;
    T maxrange = 0;
	

   
    for(int dim = 0; dim < NDIM; ++dim ) {
	
      Bounds *b = &region->bounds[dim];
	 
      T range = b->upper - b->lower;
      //if(range < 0. && blockIdx.x == 10)
      //  printf("negative range at block:%i thread:%i range:%f-%f nregions:%i\n", blockIdx.x, threadIdx.x, b->lower, b->upper, nregions);
      if( range > maxrange ) {
		maxrange = range;
		maxdim = dim;
      }
    }
	
	
    T sum[NRULES]; // NRULES is defined in quad.h as 5
    Zap(sum);		
	
    //Compute first set of permutation outside for loop to extract the Function values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * BLOCK_SIZE + threadIdx.x;
    if(pIndex < FEVAL){
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem);  
    }
    __syncthreads();
	
    //Perform operations for real f[FRAME_PER_THREAD];

    T *f = &sdata[0];
    
    if(threadIdx.x == 0){
      Result *r = &region->result;
      T *f1 = f;
      T base = *f1*2*(1 - ratio);
      T maxdiff = 0;
      int bisectdim = maxdim;
      for(int dim = 0; dim < NDIM; ++dim ){
		T *fp = f1 + 1;
		T *fm = fp + 1;
		T fourthdiff = fabs(base + ratio*(fp[0] + fm[0]) - (fp[offset] + fm[offset]));
		f1 = fm;
		if( fourthdiff > maxdiff ) {
			maxdiff = fourthdiff;
			bisectdim = dim;
		}
      }
      r->bisectdim = bisectdim;
    }
	__syncthreads();
    for(perm = 1; perm < FEVAL/BLOCK_SIZE; ++perm){
      int pIndex = perm*BLOCK_SIZE+threadIdx.x;
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem);
    }

    //Balance permutations
    __syncthreads();
    pIndex = perm*BLOCK_SIZE + threadIdx.x;
	
	
    if(pIndex < FEVAL){
      int pIndex = perm*BLOCK_SIZE + threadIdx.x;
      computePermutation<IntegT, T, NDIM>(
        d_integrand, pIndex, region->bounds, g, x, sum, constMem);  
    }
	__syncthreads();
    for(int i = 0; i < NRULES; ++i){
      sum[i] = computeReduce<T>(sum[i]);
      __syncthreads();
    }
    
    if(threadIdx.x == 0){
      Result *r = &region->result;
      for(int rul = 1; rul < NRULES - 1; ++rul ) {
		T maxerr = 0;
		for( int s = 0; s < NSETS; ++s ){
			maxerr = max(maxerr, fabs(sum[rul + 1] + __ldg(&constMem.GPUScale[s*NRULES+rul])*sum[rul])*__ldg(&constMem.GPUNorm[s*NRULES+rul]));           
		}
		sum[rul] = maxerr;
      }
      
      //if(blockIdx.x == 10 && 263 == nregions/*(parent != 0. && (vol*sum[0] + parent)  > 1.029346342330426e-07)*/)
      //  printf("id:%i %f, %f, %f, %f, %f, %f, %f, %f\n", threadIdx.x, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
      
      
      r->avg = vol*sum[0];
      r->err = vol*(
		(errcoeff[0]*sum[1] <= sum[2] && errcoeff[0]*sum[2] <= sum[3]) ?
		errcoeff[1]*sum[1] :
		errcoeff[2]*max(max(sum[1], sum[2]), sum[3]) );
      //printf("Sample : %ld %.16lf %.16lf\n",(size_t)blockIdx.x, r->avg,r->err);
    }
  }
}
