
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
    computePermutation(int pIndex, Bounds *b, T *g, T *x, T *sum){
		
		for(int dim = 0; dim < DIM; ++dim){
		  g[dim]=0;
		}
		
		int posCnt = gpuGenPermVarStart[pIndex+1]-gpuGenPermVarStart[pIndex];
		int gIndex = gpuGenPermGIndex[pIndex];
		
		T *lG = &gpuG[gIndex*DIM];
		for(int posIter = 0; posIter < posCnt; ++posIter){
		  int pos = gpuGenPos[gpuGenPermVarStart[pIndex]+posIter];
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
		for(int dim = 0; dim < DIM; ++dim ){
		  x[dim] = (.5 + g[dim])*b[dim].lower + (.5 - g[dim])*b[dim].upper;
		  T range = sBound[dim].unScaledUpper - sBound[dim].unScaledLower;
		  jacobian = jacobian*range;
		  x[dim] =  sBound[dim].unScaledLower + x[dim]*range;
		}
		
		T fun = IntegrandFunc<T>(x, DIM);
		fun = fun*jacobian;
		sdata[threadIdx.x] = fun;
		//printf("%d %lf\n",pIndex, fun);

		T *weight = &cRuleWt[gIndex*NRULES];
	  
		for(int rul = 0; rul < NRULES; ++rul ){
		  sum[rul] += fun*weight[rul];
		}
  }

  //BLOCK SIZE has to be atleast 4*DIM+1 for the first IF 
  template<typename T>
    __device__
    void
    SampleRegionBlock(int sIndex){ 
	
	//read
    Region *const region = (Region *)&sRegionPool[sIndex];
	
    T vol = ldexp(1., -region->div); // this means: 1*2^(-region->div)
    T g[DIM], x[DIM];
    int perm = 0;
	
    T ratio = Sq(gpuG[2*DIM]/gpuG[1*DIM]);
    int offset = 2*DIM;
    int maxdim = 0;
    T maxrange = 0;
	

   
    for(int dim = 0; dim < DIM; ++dim ) {
	
      Bounds *b = &region->bounds[dim];
	 
      T range = b->upper - b->lower;
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
      computePermutation<T>(pIndex, region->bounds, g, x, sum);  
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
      for(int dim = 0; dim < DIM; ++dim ){
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
	
    for(perm = 1; perm < FEVAL/BLOCK_SIZE; ++perm){
      int pIndex = perm*BLOCK_SIZE+threadIdx.x;
      computePermutation<T>(pIndex, region->bounds, g, x, sum);
    }

    //Balance permutations
    pIndex = perm*BLOCK_SIZE + threadIdx.x;
	
	
    if(pIndex < FEVAL){
      int pIndex = perm*BLOCK_SIZE + threadIdx.x;
      computePermutation<T>(pIndex, region->bounds, g, x, sum);  
    }
	
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
			maxerr = max(maxerr,fabs(sum[rul + 1] + GPUScale[s*NRULES+rul]*sum[rul])*GPUNorm[s*NRULES+rul]);
		}
		sum[rul] = maxerr;
      }
		
      r->avg = vol*sum[0];
      r->err = vol*(
		(errcoeff[0]*sum[1] <= sum[2] && errcoeff[0]*sum[2] <= sum[3]) ?
		errcoeff[1]*sum[1] :
		errcoeff[2]*max(max(sum[1], sum[2]), sum[3]) );
      //printf("Sample : %ld %.16lf %.16lf\n",(size_t)blockIdx.x, r->avg,r->err);
    }
  }
}
