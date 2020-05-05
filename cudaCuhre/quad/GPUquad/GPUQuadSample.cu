namespace quad{

	template<typename T>
    __device__ 
    T Sq(T x) {
    return x*x;
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
	
	template<typename T>
    __device__
    void
    computePermutation(int pIndex, 
					   Bounds *b, 
					   T *g, 
					   T *x, 
					   T *sum,  
					   Structures<T> *constMem){
		
		
		//if(threadIdx.x == 0 && blockIdx.x == 0)
		//	printf("computePermutation grid: %i threads & %i blocks\n", blockDim.x, gridDim.x);
		for(int dim = 0; dim < DIM; ++dim){
		  g[dim]=0;
		}
		
		int posCnt = __ldg(&constMem->_gpuGenPermVarStart[pIndex+1]) - __ldg(&constMem->_gpuGenPermVarStart[pIndex]);
		int gIndex = constMem->_gpuGenPermGIndex[pIndex];
		
		for(int posIter = 0; posIter < posCnt; ++posIter){
		  int pos = __ldg(&constMem->_gpuGenPos[__ldg(&constMem->_gpuGenPermVarStart[pIndex])+posIter]);
		  int absPos = abs(pos);
		  if(pos == absPos){
			  g[absPos-1] =  __ldg(&constMem->_gpuG[gIndex*DIM+posIter]);
		  }
		  else{
			  g[absPos-1] =  -__ldg(&constMem->_gpuG[gIndex*DIM+posIter]);
			}
		}
		
		/*sBound[0].unScaledLower = .5;sBound[0].unScaledUpper = 1;
		sBound[1].unScaledLower = 0;;sBound[1].unScaledUpper = 1 ;
		sBound[2].unScaledLower = 0;sBound[2].unScaledUpper = 1;
		sBound[3].unScaledLower = 0;sBound[3].unScaledUpper = 1;
		sBound[4].unScaledLower = 0;sBound[4].unScaledUpper = 1;
		sBound[5].unScaledLower = 0;sBound[5].unScaledUpper = 1;*/
		
		/*sBound[0].unScaledLower = .5;sBound[0].unScaledUpper = .75;
		sBound[1].unScaledLower = 0;;sBound[1].unScaledUpper = 1 ;
		sBound[2].unScaledLower = 0;sBound[2].unScaledUpper = 1;
		sBound[3].unScaledLower = 0;sBound[3].unScaledUpper = 1;
		sBound[4].unScaledLower = 0;sBound[4].unScaledUpper = 1;
		sBound[5].unScaledLower = 0;sBound[5].unScaledUpper = 1;*/
		
		//sBound is shared memory
		T jacobian = 1;
		for(int dim = 0; dim < DIM; ++dim ){
		  x[dim] = (.5 + g[dim])*b[dim].lower + (.5 - g[dim])*b[dim].upper;
		  T range = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;
		  jacobian = jacobian*range;
		  x[dim] =  sBound[dim].unScaledLower + x[dim]*range;
		}
		
		T fun = IntegrandFunc<T>(x, DIM);
		//printf("[%i] (%i) Computing integral at x:%.12f at %f, %f, %f, %f, %f, %f\n", blockIdx.x, threadIdx.x, fun, x[0], x[1], x[2], x[3], x[4], x[5]);
		//commented out by Ioannis
		fun = fun*jacobian;
		sdata[threadIdx.x] = fun;
	
		//added by Ioannis, effectively changed the order of where this is changing
		//fun = fun*jacobian;
		//T *weight = &constMem->_cRuleWt[gIndex*NRULES];
	  
		for(int rul = 0; rul < NRULES; ++rul ){
		  //sum[rul] += fun*weight[rul];
		  sum[rul] += fun*__ldg(&constMem->_cRuleWt[gIndex*NRULES+rul]);
		}
  }
	
  //BLOCK SIZE has to be atleast 4*DIM+1 for the first IF 
	template<typename T>
    __device__
    void
    SampleRegionBlock(int sIndex,  
					  Structures<T> *constMem,
					  int FEVAL,
					  int NSETS){ 
	
	//read
    Region *const region = (Region *)&sRegionPool[sIndex];
	
    T vol = ldexp(1., -region->div); // this means: 1*2^(-region->div)
    T g[DIM], x[DIM];
    int perm = 0;
	
    T ratio = Sq(__ldg(&constMem->_gpuG[2*DIM])/__ldg(&constMem->_gpuG[1*DIM]));
    int offset = 2*DIM;
    int maxdim = 0;
    T maxrange = 0;
	
	//set dimension range
    for(int dim = 0; dim < DIM; ++dim ) {
	
      Bounds *b = &region->bounds[dim];
      T range = b->upper - b->lower;
      if( range > maxrange ) {
		maxrange = range;
		maxdim = dim;
      }
    }
	
    T sum[NRULES]; 
    Zap(sum);		
	
    //Compute first set of permutation outside for loop to extract the Function values for the permutation used to compute
    // fourth dimension
    int pIndex = perm * BLOCK_SIZE + threadIdx.x;
	//printf("%lu\n", FEVAL); 
	 __syncthreads();
    if(pIndex < FEVAL){
		//printf("[%i] Thread %i doing 1st permutation\n", blockIdx.x, threadIdx.x);
      computePermutation<T>(pIndex, region->bounds, g, x, sum, constMem);  
    }
	
    __syncthreads();
	
    //Perform operations for real f[FRAME_PER_THREAD];
	
    T *f = &sdata[0];
	__syncthreads();
	//if(threadIdx.x ==0)
		//for(int ii=0; ii<15; ii++)
		{
			//printf("[%i] sData[%i]:%.12f\n", blockIdx.x, ii, sdata[ii]);
		}
	
    if(threadIdx.x == 0){
      Result *r = &region->result;
      T *f1 = f;
      T base = *f1*2*(1 - ratio);
      T maxdiff = 0;
      int bisectdim = maxdim;
      for(int dim = 0; dim < DIM; ++dim ){
		T *fp = f1 + 1;
		T *fm = fp + 1;
		//if(threadIdx.x ==0)
		//	printf("[%i] base:%.12f  ratio:%.12f  fp[0]:%.12f  fm[0]:%.12f fp[%i]:%.12f fm[%i]:%.12f\n", blockIdx.x, base, ratio, fp[0], fm[0], offset, fp[offset], offset, fm[offset]);
		T fourthdiff = fabs(base + ratio*(fp[0] + fm[0]) - (fp[offset] + fm[offset]));
		f1 = fm;
		//printf("[%i] fourthdiff:%.12f at dim:%i\n", blockIdx.x, fourthdiff, dim);
		if( fourthdiff > maxdiff ) {
			maxdiff = fourthdiff;
			bisectdim = dim;
		}
      }
      r->bisectdim = bisectdim;
	  //printf("[%i] bisectDim:%i\n", blockIdx.x, bisectdim);
    }
	__syncthreads();
	
	//value is the set number of required function evaluations per region, not how many there have been so far or anythign like that
	
    for(perm = 1; perm < FEVAL/BLOCK_SIZE; ++perm){
      int pIndex = perm*BLOCK_SIZE+threadIdx.x;
	  //printf("[%i] Thread %i doing 2nd permutation\n", blockIdx.x, threadIdx.x);
      computePermutation<T>(pIndex, region->bounds, g, x, sum, constMem);
    }
	
    //Balance permutations
    pIndex = perm*BLOCK_SIZE + threadIdx.x;
    if(pIndex < FEVAL){
      int pIndex = perm*BLOCK_SIZE + threadIdx.x;
	 // printf("[%i] Thread %i doing 3rd permutation\n", blockIdx.x, threadIdx.x);
      computePermutation<T>(pIndex, region->bounds, g, x, sum, constMem);  
    }
	
	//FIRST TIME REDUCTION HAPPENS WITH SUM
    for(int i = 0; i < NRULES; ++i)
      sum[i] = computeReduce<T>(sum[i]);
  
    if(threadIdx.x == 0){
      Result *r = &region->result;
      // Search for the null rule, in the linear space spanned by two
      //   successive null rules in our sequence, which gives the greatest
      //   error estimate among all normalized (1-norm) null rules in this
      //   space. 
      for(int rul = 1; rul < NRULES - 1; ++rul ) {
		T maxerr = 0;
		for( int s = 0; s < NSETS; ++s ){
			maxerr = MAX(maxerr, fabs(sum[rul + 1] + constMem->_GPUScale[s*NRULES+rul]*sum[rul])*constMem->_GPUNorm[s*NRULES+rul]);
		}
		sum[rul] = maxerr;
      }
		
      r->avg = vol*sum[0];
      r->err = vol*(
		(errcoeff[0]*sum[1] <= sum[2] && errcoeff[0]*sum[2] <= sum[3]) ?
		errcoeff[1]*sum[1] :
		errcoeff[2]*MAX(MAX(sum[1], sum[2]), sum[3]) );
      //printf("Sample : %ld %.16lf %.16lf\n",(size_t)blockIdx.x, r->avg,r->err);
	  if(blockIdx.x<5)
		printf("[%i] Phase 1 unrefined error %.12f +- %.12f\n", blockIdx.x, r->avg, r->err);
    }
  }
}
