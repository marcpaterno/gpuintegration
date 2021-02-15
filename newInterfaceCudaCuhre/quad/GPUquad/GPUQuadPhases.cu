#include "GPUQuadSample.cu"

namespace quad{

    template <typename IntegT, typename T, int NDIM>
    __device__
    void
    INIT_REGION_POOL(IntegT* d_integrand, T *dRegions, T *dRegionsLength, size_t numRegions,  Region<NDIM> sRegionPool[], const Structures<T>& constMem, int FEVAL,  int NSETS){
    size_t index = blockIdx.x;
    
    if(threadIdx.x == 0){
      for(int dim = 0; dim < NDIM; ++dim){
		T lower = dRegions[dim * numRegions + index];
		sRegionPool[threadIdx.x].bounds[dim].lower = 0;
		sRegionPool[threadIdx.x].bounds[dim].upper = 1;

		sBound[dim].unScaledLower = lower;
		sBound[dim].unScaledUpper = lower + dRegionsLength[dim * numRegions + index];
        
		sRegionPool[threadIdx.x].div = 0;
      }
    }
    __syncthreads();
    SampleRegionBlock<IntegT, T, NDIM>(d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool);
    __syncthreads();
  }

  template <typename IntegT, typename T, int NDIM>
    __global__
    void 
    INTEGRATE_GPU_PHASE1(IntegT* d_integrand, T *dRegions, T *dRegionsLength, size_t numRegions, T *dRegionsIntegral, T *dRegionsError, int *activeRegions, int *subDividingDimension, T epsrel, T epsabs, Structures<T> constMem,
                       int FEVAL,
                       int NSETS){
    T ERR = 0, RESULT = 0;
    typedef  Region<NDIM> Region;
    __shared__ Region sRegionPool[SM_REGION_POOL_SIZE];
    
    int fail = 0;
    INIT_REGION_POOL<IntegT, T, NDIM>(d_integrand, dRegions, dRegionsLength, numRegions,  sRegionPool, constMem, FEVAL, NSETS);

    if(threadIdx.x == 0){
      ERR = sRegionPool[threadIdx.x].result.err;
      RESULT = sRegionPool[threadIdx.x].result.avg;
      T ratio  = ERR/MaxErr(RESULT, epsrel, epsabs);
      int fourthDiffDim = sRegionPool[threadIdx.x].result.bisectdim;
      dRegionsIntegral[gridDim.x + blockIdx.x] = RESULT;
      dRegionsError[gridDim.x + blockIdx.x] = ERR;

      if(ratio > 1){
      	fail = 1;
      	ERR = 0;
      	RESULT = 0;
      }
      activeRegions[blockIdx.x] = fail;
      subDividingDimension[blockIdx.x] = fourthDiffDim;
      dRegionsIntegral[blockIdx.x] = RESULT;
      dRegionsError[blockIdx.x]=ERR;    
    }  
  }



  ////PHASE 2 Procedures Starts


  template<typename T, int NDIM>
    __device__
    void
    ComputeErrResult(T &ERR, T &RESULT, Region<NDIM> sRegionPool[]){
    /*sdata[threadIdx.x] = sRegionPool[threadIdx.x].result.err;
    sdata[blockDim.x + threadIdx.x] = sRegionPool[threadIdx.x].result.avg;
    __syncthreads();

    // contiguous range pattern
    for(size_t offset = size / 2; offset > 0; offset >>= 1){
      if(threadIdx.x < offset){
	sdata[threadIdx.x] += sdata[threadIdx.x + offset];
	sdata[blockDim.x + threadIdx.x] += sdata[blockDim.x + threadIdx.x + offset];
      }
      __syncthreads();
    }
    */
    if(threadIdx.x == 0){
      ERR = sRegionPool[threadIdx.x].result.err;
      RESULT = sRegionPool[threadIdx.x].result.avg;
    }
    __syncthreads();
  }

    template <typename IntegT, typename T, int NDIM>
    __device__ int
    INIT_REGION_POOL(IntegT* d_integrand, T *dRegions, T *dRegionsLength, int *subDividingDimension,  size_t numRegions, Region<NDIM> sRegionPool[],  Region<NDIM>*& gPool, Structures<T>& constMem, int FEVAL, int NSETS){
        
    typedef Region<NDIM> Region;
    size_t intervalIndex = blockIdx.x;
    int idx = 0;
	
	//idx<0 always? SM_R = 128 (quad.h) BLOCK_SIZE=256
    for(; idx < SM_REGION_POOL_SIZE/BLOCK_SIZE; ++idx){
		
      int index = idx*BLOCK_SIZE + threadIdx.x;
      sRegionPool[index].div = 0;
      sRegionPool[index].result.err = 0;
      sRegionPool[index].result.avg = 0;
      sRegionPool[index].result.bisectdim = 0;
  
      for(int dim = 0; dim < NDIM; ++dim){
		sRegionPool[index].bounds[dim].lower = 0;
		sRegionPool[index].bounds[dim].upper = 0;
      }
    }
	
    int index = idx*BLOCK_SIZE + threadIdx.x; //essentially threadIdx.x
    if(index < SM_REGION_POOL_SIZE){
	  
      sRegionPool[index].div = 0;
      sRegionPool[index].result.err = 0;
      sRegionPool[index].result.avg = 0;
      sRegionPool[index].result.bisectdim = 0;
  
      for(int dim = 0; dim < NDIM; ++dim){
		sRegionPool[index].bounds[dim].lower = 0;
		sRegionPool[index].bounds[dim].upper = 0;
      }      
    }
    
	//gets unscaled lower and upper bounds for region
    if(threadIdx.x == 0){
      for(int dim = 0; dim < NDIM; ++dim){
		 
		sRegionPool[threadIdx.x].bounds[dim].lower = 0;
		sRegionPool[threadIdx.x].bounds[dim].upper = 1;
    	T lower = dRegions[dim * numRegions + intervalIndex];
		sBound[dim].unScaledLower = lower;
		sBound[dim].unScaledUpper = lower + dRegionsLength[dim * numRegions + intervalIndex];
      }    
    }
    
    __syncthreads();
	 
    SampleRegionBlock<IntegT, T, NDIM>(d_integrand, 0, constMem, FEVAL, NSETS, sRegionPool);
    
    if(threadIdx.x == 0){
      gPool = (Region *)malloc(sizeof(Region)*(SM_REGION_POOL_SIZE/2));
      gRegionPoolSize = (SM_REGION_POOL_SIZE/2);//BLOCK_SIZE;
      
      if(gPool == NULL)
        printf("gPool not allocated right\n");
    }
	
    __syncthreads();
	
    for(idx = 0; idx < (SM_REGION_POOL_SIZE/2)/BLOCK_SIZE; ++idx){
      int index = idx*BLOCK_SIZE + threadIdx.x;
      gRegionPos[index]=index;
      gPool[index]=sRegionPool[index];
    }
	
    index = idx*BLOCK_SIZE + threadIdx.x;
    if(index < (SM_REGION_POOL_SIZE/2)){
      gRegionPos[index]=index;
      gPool[index]=sRegionPool[index];
    }
    return 1;
  }
  
  template <class T>
    __device__
    void swap ( T& a, T& b ){
		T c(a);
		a=b; b=c;
  }
  
  template<int NDIM>
  __device__
  void CheckRegion(Region<NDIM> r, int id, int indexSource, int indexDest){
    for(int i=0; i<NDIM; i++){
            if(r.bounds[i].lower >= r.bounds[i].upper)
                printf("found problem at dim %i at block %i thread %i funcID:%i bounds:%f, %f, copy indexes %i -> %i val:%e +- %e\n", i, blockIdx.x, threadIdx.x, id, r.bounds[i].lower, r.bounds[i].upper,indexSource, indexDest, r.result.avg, r.result.err);
    }
  }
  
  template<typename T, int NDIM>
    __device__
    void
    INSERT_GLOBAL_STORE(Region<NDIM> *sRegionPool, Region<NDIM>*& gRegionPool,  Region<NDIM>*& gPool, int gpuId){
        
		typedef Region<NDIM> Region;
		if(threadIdx.x == 0){
			gPool = (Region *)malloc(sizeof(Region)*(gRegionPoolSize+((size_t)SM_REGION_POOL_SIZE/2)));
			if(gPool == NULL){
				printf("Failed to malloc at block:%i threadIndex:%i gpu:%i currentSize:%lu requestedSize:%lu\n", blockIdx.x, threadIdx.x, gpuId, gRegionPoolSize , gRegionPoolSize+((size_t)SM_REGION_POOL_SIZE/2));
			}
		}
		__syncthreads();
		
		//Copy existing global regions into newly allocated spaced
		int iterationsPerThread = 0;
		for(iterationsPerThread = 0; iterationsPerThread < gRegionPoolSize/BLOCK_SIZE; ++iterationsPerThread){
		  size_t dataIndex = iterationsPerThread*BLOCK_SIZE + threadIdx.x;
		 			  
		  gPool[dataIndex] = gRegionPool[dataIndex];
		  __syncthreads(); 
		}
		
		size_t dataIndex = iterationsPerThread*BLOCK_SIZE + threadIdx.x;
		if(dataIndex < gRegionPoolSize){
		  gPool[dataIndex] = gRegionPool[dataIndex];
          
		}
		
	
		//Fill the previous occupied postion in global memory by half of shared memory regions
		for(iterationsPerThread = 0; iterationsPerThread < (SM_REGION_POOL_SIZE/2)/BLOCK_SIZE; ++iterationsPerThread){
		  size_t index = iterationsPerThread*BLOCK_SIZE+threadIdx.x;
          
          
          for(int i=0; i<NDIM; i++)
          {
            gPool[gRegionPoolSize + index].bounds[i].lower = sRegionPool[(SM_REGION_POOL_SIZE/2) + index].bounds[i].lower;  
            gPool[gRegionPoolSize + index].bounds[i].upper = sRegionPool[(SM_REGION_POOL_SIZE/2) + index].bounds[i].upper;  
            
            if(sRegionPool[(SM_REGION_POOL_SIZE/2) + index].bounds[i].lower >= sRegionPool[(SM_REGION_POOL_SIZE/2) + index].bounds[i].upper)
                printf("Block %i shared memory bad bounds\n", blockIdx.x);
            
             if(gPool[gRegionPos[index]].bounds[i].lower >= gPool[gRegionPos[index]].bounds[i].upper)
                printf("Block %i global memory bad bounds\n", blockIdx.x);
            
            gPool[gRegionPos[index]].bounds[i].lower = sRegionPool[index].bounds[i].lower;
            gPool[gRegionPos[index]].bounds[i].upper = sRegionPool[index].bounds[i].upper;

          }
          
		  gPool[gRegionPos[index]] = sRegionPool[index];
		  gPool[gRegionPoolSize + index] = sRegionPool[(SM_REGION_POOL_SIZE/2) + index];
          
          //CheckRegion(sRegionPool[index],       1, index, gRegionPos[index]);
          //CheckRegion(gPool[gRegionPos[index]], 2, index, gRegionPos[index]);
          
          //CheckRegion(sRegionPool[(SM_REGION_POOL_SIZE/2) + index], 1, (SM_REGION_POOL_SIZE/2) + index, gRegionPoolSize + index);
          //CheckRegion(gPool[gRegionPoolSize + index],               8, (SM_REGION_POOL_SIZE/2) + index, gRegionPoolSize + index);
          
          
       
		}
		
        __syncthreads();//hope it's this one

		int index = iterationsPerThread*BLOCK_SIZE+threadIdx.x;
		if(index < (SM_REGION_POOL_SIZE/2)){
		  //int index = iterationsPerThread*BLOCK_SIZE+threadIdx.x;
		  gPool[gRegionPos[index]] = sRegionPool[index];
		  gPool[gRegionPoolSize + index] = sRegionPool[(SM_REGION_POOL_SIZE/2) + index];
          
          
          for(int i=0; i<NDIM; i++)
          {
            gPool[gRegionPoolSize + index].bounds[i].lower = sRegionPool[(SM_REGION_POOL_SIZE/2) + index].bounds[i].lower;  
            gPool[gRegionPoolSize + index].bounds[i].upper = sRegionPool[(SM_REGION_POOL_SIZE/2) + index].bounds[i].upper;  
            
            gPool[gRegionPos[index]].bounds[i].lower = sRegionPool[index].bounds[i].lower;
            gPool[gRegionPos[index]].bounds[i].upper = sRegionPool[index].bounds[i].upper;
            
            //if(sRegionPool[(SM_REGION_POOL_SIZE/2) + index].bounds[i].lower >= sRegionPool[(SM_REGION_POOL_SIZE/2) + index].bounds[i].upper)
            //    printf("Block %i shared memory bad bounds\n", blockIdx.x);

            //if(gPool[gRegionPos[index]].bounds[i].lower >= gPool[gRegionPos[index]].bounds[i].upper)
            //    printf("Block %i global memory bad bounds\n", blockIdx.x);

          }
          
          
          //CheckRegion(sRegionPool[index], 1, index, gRegionPos[index]);
          //CheckRegion(sRegionPool[(SM_REGION_POOL_SIZE/2) + index], 1, (SM_REGION_POOL_SIZE/2) + index, gRegionPoolSize + index);
          
         // CheckRegion(gPool[gRegionPos[index]], 2, index, gRegionPos[index]);
          //CheckRegion(gPool[gRegionPoolSize + index], 8, (SM_REGION_POOL_SIZE/2) + index, gRegionPoolSize + index);
		}

		__syncthreads();
		if(threadIdx.x == 0){
		  gRegionPoolSize = gRegionPoolSize+(SM_REGION_POOL_SIZE/2);
		  free(gRegionPool);
		}
		__syncthreads();
  }

  template<typename T>
    __device__
    void
    EXTRACT_MAX(T *serror, size_t *serrorPos, size_t gSize){
		
    for(size_t offset = gSize/2; offset > 0; offset >>= 1 ){
      int idx = 0;
      for(idx = 0; idx < offset/BLOCK_SIZE; ++idx){
		size_t index = idx*BLOCK_SIZE+threadIdx.x;
		if(index < offset){
			if(serror[index] < serror[index+offset]){
				swap(serror[index], serror[index+offset]);
				swap(serrorPos[index], serrorPos[index+offset]);
			}
			//printf("%ld %ld\n",index, index+offset);
		}
      }
      size_t index = idx*BLOCK_SIZE+threadIdx.x;
      if(index < offset){
        if(serror[index] < serror[index+offset]){
          swap(serror[index], serror[index+offset]);
          swap(serrorPos[index], serrorPos[index+offset]);
        }
      }
      __syncthreads();
    }
  }
  
  template<typename T, int NDIM>
    __device__
    void
    EXTRACT_TOPK(Region<NDIM> *sRegionPool, Region<NDIM> *gRegionPool, Region<NDIM>* gPool){
        typedef Region<NDIM> Region;
    
        T *sarray = (T *)&sRegionPool[0];

        if(threadIdx.x == 0){
          //T *sarray = (T *)&sRegionPool[0];

          if((gRegionPoolSize*sizeof(T) + gRegionPoolSize*sizeof(size_t)) < sizeof(Region) * SM_REGION_POOL_SIZE){
            serror = &sarray[0];
            serrorPos = (size_t *)&sarray[gRegionPoolSize];
          }
          else{
            serror = (T *)malloc(sizeof(T) * gRegionPoolSize);
            serrorPos = (size_t *)malloc(sizeof(size_t) * gRegionPoolSize);
            if(serror == NULL || serrorPos == NULL)
                printf("problem with serror allocation\n");
          }
        }
        __syncthreads();
	
        int offset = 0;
        for(offset = 0; (offset < MAX_GLOBALPOOL_SIZE/BLOCK_SIZE) && (offset < gRegionPoolSize/BLOCK_SIZE); offset++){
          size_t regionIndex = offset*BLOCK_SIZE + threadIdx.x;
          serror[regionIndex] = gRegionPool[regionIndex].result.err;
          serrorPos[regionIndex] = regionIndex;
        }
        
        size_t regionIndex = offset*BLOCK_SIZE + threadIdx.x;
        if(regionIndex < gRegionPoolSize){
          serror[regionIndex] = gRegionPool[regionIndex].result.err;
          serrorPos[regionIndex] = regionIndex;
        }

        __syncthreads();
        for(int k = 0; k < (SM_REGION_POOL_SIZE/2); ++k){
          EXTRACT_MAX<T>(&serror[k], &serrorPos[k], gRegionPoolSize-k);
        }
        
        int iterationsPerThread = 0;
        for(iterationsPerThread = 0; iterationsPerThread < (SM_REGION_POOL_SIZE/2)/BLOCK_SIZE; ++iterationsPerThread){
          int index = iterationsPerThread*BLOCK_SIZE + threadIdx.x;
          size_t pos = serrorPos[index];
          gRegionPos[index] = pos;
        }
        int index = iterationsPerThread*BLOCK_SIZE + threadIdx.x;
        if(index < (SM_REGION_POOL_SIZE/2)){
          size_t pos = serrorPos[index];
          gRegionPos[index] = pos;
        }

        __syncthreads();

        if(threadIdx.x == 0){
          if(2*gRegionPoolSize*sizeof(T) >= sizeof(Region) * SM_REGION_POOL_SIZE){
            free(serror);
            free(serrorPos);
          }
        }
        __syncthreads();   
        
        //Copy top K into SM and reset the remaining
        for(iterationsPerThread = 0; iterationsPerThread < (SM_REGION_POOL_SIZE/2)/BLOCK_SIZE; ++iterationsPerThread){
          int index = iterationsPerThread*BLOCK_SIZE + threadIdx.x;
          sRegionPool[index] = gPool[gRegionPos[index]];
          //CheckRegion(sRegionPool[index], 3, gRegionPos[index], index);
          sRegionPool[(SM_REGION_POOL_SIZE/2) + index].result.err = -INFTY;
          sRegionPool[(SM_REGION_POOL_SIZE/2) + index].result.avg = 0;
          sRegionPool[(SM_REGION_POOL_SIZE/2) + index].div = 0;
        }
	
        index = iterationsPerThread*BLOCK_SIZE + threadIdx.x;
        if(index < (SM_REGION_POOL_SIZE/2)){
          sRegionPool[index] = gPool[gRegionPos[index]];
          //CheckRegion(sRegionPool[index], 3, gRegionPos[index], index);
          sRegionPool[(SM_REGION_POOL_SIZE/2) + index].result.err = -INFTY;
          sRegionPool[(SM_REGION_POOL_SIZE/2) + index].result.avg = 0;
          sRegionPool[(SM_REGION_POOL_SIZE/2) + index].div = 0;
        }
  }


  template<typename T, int NDIM>
    __device__
    size_t
    EXTRACT_MAX(Region<NDIM> *sRegionPool, Region<NDIM>*& gRegionPool, size_t sSize, int gpuId, Region<NDIM>*& gPool){
    
        typedef Region<NDIM> Region;
		if(sSize == SM_REGION_POOL_SIZE){
		  INSERT_GLOBAL_STORE<T, NDIM>(sRegionPool, gRegionPool, gPool, gpuId);
		  __syncthreads();
			
		  gRegionPool = gPool;
		  EXTRACT_TOPK<T>(sRegionPool, gRegionPool, gPool);
		  sSize = (SM_REGION_POOL_SIZE/2);
		  __syncthreads();
		}
		
		for(size_t offset = (SM_REGION_POOL_SIZE/2); offset > 0; offset >>= 1 ){
		  int idx = 0;
		  for(idx = 0; idx < offset/BLOCK_SIZE; ++idx){
			size_t index = idx*BLOCK_SIZE+threadIdx.x;	
			if(index < offset){
				Region *r1 = &sRegionPool[index];
				Region *r2 = &sRegionPool[index + offset];
				if(r1->result.err < r2->result.err ){
                    //CheckRegion(sRegionPool[index], 4, sSize, sSize);
                    //CheckRegion(sRegionPool[index + offset], 4, sSize, sSize);
					swap<Region>(sRegionPool[index], sRegionPool[offset + index]);
				}
			}
		  }
		  
		  size_t index = idx*BLOCK_SIZE+threadIdx.x;
		  if(index < offset){
			Region *r1 = &sRegionPool[index];
			Region *r2 = &sRegionPool[index + offset];
			if(r1->result.err < r2->result.err ){
                //CheckRegion(sRegionPool[index], 4, sSize, sSize);
                //CheckRegion(sRegionPool[index + offset], 4, sSize, sSize);
				swap<Region>(sRegionPool[index], sRegionPool[offset + index]);
			}
		  }
		  __syncthreads();
		}

    return sSize;
  }


    template <typename IntegT, typename T, int NDIM>
    __global__
    void
    BLOCK_INTEGRATE_GPU_PHASE2(IntegT* d_integrand, T *dRegions, T *dRegionsLength, size_t numRegions, T *dRegionsIntegral, T *dRegionsError, int *dRegionsNumRegion, int *activeRegions, int *subDividingDimension, T epsrel, T epsabs, int gpuId, Structures<T> constMem, int FEVAL,  int NSETS){
        
		typedef  Region<NDIM> Region;
        __shared__ Region* gPool;
        __shared__ Region sRegionPool[SM_REGION_POOL_SIZE];
		Region *gRegionPool = 0;	
       
		int sRegionPoolSize = INIT_REGION_POOL<IntegT, T, NDIM>(d_integrand, dRegions, dRegionsLength, subDividingDimension,  numRegions, sRegionPool, gPool, constMem, FEVAL, NSETS);
		ComputeErrResult<T>(ERR, RESULT, sRegionPool);
		
		__syncthreads();
		int nregions = sRegionPoolSize; //is only 1 at this point
		
		for(; (nregions < MAX_GLOBALPOOL_SIZE) && (nregions == 1 || ERR > MaxErr(RESULT, epsrel, epsabs)); ++nregions ){
			
			gRegionPool = gPool;
			sRegionPoolSize = EXTRACT_MAX<T>(sRegionPool, gRegionPool, sRegionPoolSize, gpuId, gPool);
			//CheckRegion(sRegionPool[0], -1, sRegionPoolSize, nregions);
			Region *RegionLeft, *RegionRight;
			Result result;
			
			if(threadIdx.x == 0){		
				Bounds *bL, *bR;
				Region *R = &sRegionPool[0];
				result.err = R->result.err;
				result.avg = R->result.avg;
				result.bisectdim = R->result.bisectdim;
				
				int bisectdim = result.bisectdim;
 
				RegionLeft = R;
				RegionRight = &sRegionPool[sRegionPoolSize];
				
				bL = &RegionLeft->bounds[bisectdim];
				bR = &RegionRight->bounds[bisectdim];
				
				RegionRight->div = ++RegionLeft->div;
				for(int dim = 0; dim < NDIM; ++dim){
					RegionRight->bounds[dim].lower = RegionLeft->bounds[dim].lower;
					RegionRight->bounds[dim].upper = RegionLeft->bounds[dim].upper;
				}
				//Subdivide the chosen axis
				bL->upper = bR->lower = 0.5 * (bL->lower + bL->upper);	
			}
			
			sRegionPoolSize++;
			
			__syncthreads();
			SampleRegionBlock<IntegT, T, NDIM>(d_integrand, 0, constMem, FEVAL,  NSETS,  sRegionPool, nregions);
			__syncthreads();
			SampleRegionBlock<IntegT, T, NDIM>(d_integrand, sRegionPoolSize-1, constMem, FEVAL,  NSETS,  sRegionPool, /*(&RegionLeft->result)->avg - result.avg*/ nregions);
			__syncthreads();
			
			if(threadIdx.x == 0){
                //CheckRegion(sRegionPool[0], 0,sRegionPoolSize, nregions);
                //CheckRegion(sRegionPool[sRegionPoolSize-1], 0, sRegionPoolSize, nregions);
                
				Result *rL = &RegionLeft->result;
				Result *rR = &RegionRight->result;
                
				T diff = rL->avg + rR->avg - result.avg;
                double left = rL->avg;
                double right = rR->avg;
				diff = fabs(.25*diff);
				T err = rL->err + rR->err;
                
				if( err > 0 ) {
				  T c = 1 + 2*diff/err;
				  rL->err *= c;
				  rR->err *= c;
				}
                
				rL->err += diff;
				rR->err += diff;
                    
				ERR += rL->err + rR->err - result.err;
				RESULT +=  rL->avg + rR->avg - result.avg;   
			}
			__syncthreads();
		}	
		
		if(threadIdx.x == 0){
            int isActive = ERR > MaxErr(RESULT, epsrel, epsabs);
		  
			if(ERR > (1e+10)){
				printf("Bad region at block:%i\n", blockIdx.x);
				
				RESULT = 0.0;
				ERR = 0.0;
				isActive = 1;
			}
			
            //if(blockIdx.x == 10)
            //    printf("it:%i block = %.15e, %.15e nregions:%i sRegionPoolSize:%i\n", nregions, RESULT, ERR, nregions, sRegionPoolSize);

			activeRegions[blockIdx.x] = isActive;
			dRegionsIntegral[blockIdx.x] = RESULT;
			dRegionsError[blockIdx.x] = ERR;
			dRegionsNumRegion[blockIdx.x] = nregions;		
			free(gPool);
		}
  }
}
