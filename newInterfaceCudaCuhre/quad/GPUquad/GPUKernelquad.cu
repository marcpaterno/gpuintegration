
#include "GPUQuadPhases.cu"
#include "../util/Volume.cuh"

namespace quad{

//===========
//FOR DEBUGGING

/*bool cudaMemoryTest()
{
	bool status = false;
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
    cudaMalloc((int**)&d_a, bytes);

    memset(h_a, 0, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    //status = true;
	return true;
}*/

//==========

  __constant__ size_t dFEvalPerRegion;
  template<typename T> 
    __global__
    void
    generateInitialRegions(T *dRegions, T *dRegionsLength, size_t numRegions,  T *newRegions, T *newRegionsLength, size_t newNumOfRegions, int numOfDivisionsPerRegionPerDimension, int NDIM){
    extern __shared__ T slength[];
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x < NDIM){
      slength[threadIdx.x] = dRegionsLength[threadIdx.x]/numOfDivisionsPerRegionPerDimension;
    }
    __syncthreads();

    if(threadId < newNumOfRegions){
      size_t interval_index = threadId / pow((T)numOfDivisionsPerRegionPerDimension, (T)NDIM);
      size_t local_id = threadId %  (size_t)pow((T)numOfDivisionsPerRegionPerDimension, (T)NDIM);
      for(int dim = 0; dim < NDIM; ++dim){
        size_t id = (size_t)(local_id/pow((T)numOfDivisionsPerRegionPerDimension, (T)dim)) % numOfDivisionsPerRegionPerDimension;
        newRegions[newNumOfRegions*dim + threadId]       = dRegions[numRegions*dim + interval_index] + id*slength[dim];
        newRegionsLength[newNumOfRegions*dim + threadId] = slength[dim];;
      }
    }
    
  }

  template<typename T, int NDIM> 
    __global__
    void
    alignRegions(T *dRegions, T *dRegionsLength, int *activeRegions, int *subDividingDimension, int *scannedArray, T *newActiveRegions, T *newActiveRegionsLength, int *newActiveRegionsBisectDim, size_t numRegions, size_t newNumRegions, int numOfDivisionOnDimension){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid <  numRegions && activeRegions[tid] == 1){
      size_t interval_index = scannedArray[tid];
      for(int i = 0 ; i < NDIM; ++i){
	newActiveRegions[i*newNumRegions + interval_index] = dRegions[i*numRegions + tid];
	newActiveRegionsLength[i*newNumRegions + interval_index] = dRegionsLength[i*numRegions + tid];
      }
      for(int i = 0; i < numOfDivisionOnDimension; ++i){
	newActiveRegionsBisectDim[i*newNumRegions + interval_index] = subDividingDimension[tid];
      }
    }
  }

  template<typename T, int NDIM>
    __global__
    void
    divideIntervalsGPU(T *genRegions, T *genRegionsLength, T *activeRegions, T *activeRegionsLength, int *activeRegionsBisectDim, size_t numActiveRegions, int numOfDivisionOnDimension){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < numActiveRegions){
      int bisectdim = activeRegionsBisectDim[tid];
      size_t data_size = numActiveRegions*numOfDivisionOnDimension;
      for(int i = 0; i < numOfDivisionOnDimension; ++i){
	for(int dim = 0; dim < NDIM; ++dim){
	  genRegions[i*numActiveRegions + dim*data_size + tid]=activeRegions[dim*numActiveRegions + tid];
	  genRegionsLength[i*numActiveRegions + dim*data_size + tid]=activeRegionsLength[dim*numActiveRegions + tid];
	}
      }
      for(int i = 0; i < numOfDivisionOnDimension; ++i){
	T interval_length = activeRegionsLength[bisectdim*numActiveRegions + tid]/numOfDivisionOnDimension;
	genRegions[bisectdim*data_size + i*numActiveRegions + tid] = activeRegions[bisectdim*numActiveRegions + tid] + i*interval_length;
	genRegionsLength[i*numActiveRegions + bisectdim*data_size + tid] = interval_length;
      }
    }
  }
    
  template<typename T, int NDIM>
    class GPUKernelCuhre{
    T *dRegions;
    T *dRegionsLength;
    T *hRegions;
    T *hRegionsLength;
    
    T* highs;
    T* lows;
    
    int KEY, VERBOSE;
    size_t numRegions, numFunctionEvaluations;
    size_t fregions;
    size_t fEvalPerRegion;
    HostMemory<T> Host;
    DeviceMemory<T> Device;
    QuadRule<T, NDIM> Rule;
	Structures<T> constMem;
    
    int NUM_DEVICES ;
	int depthBeingProcessed;
    int lastPhase; 
    
	//Debug Msg
    char msg[256];
    std::ostream &log;
 
  public:

	GPUKernelCuhre(std::ostream &logerr=std::cout):log(logerr){
      numRegions = 0;
      numFunctionEvaluations = 0;
      KEY = 0;
      lastPhase = 1;
      fregions = 0;
    }

    ~GPUKernelCuhre(){
      if(VERBOSE){
        sprintf(msg, "GPUKerneCuhre Destructur");
        Println(log, msg);
      }
      QuadDebug(Device.ReleaseMemory(dRegions));
      QuadDebug(Device.ReleaseMemory(dRegionsLength));
      Host.ReleaseMemory(hRegions);
      Host.ReleaseMemory(hRegionsLength);
      QuadDebug(cudaFree(constMem.gpuG));
      QuadDebug(cudaFree(constMem.cRuleWt));
      QuadDebug(cudaFree(constMem.GPUScale));
      QuadDebug(cudaFree(constMem.GPUNorm));
      QuadDebug(cudaFree(constMem.gpuGenPos));
      QuadDebug(cudaFree(constMem.gpuGenPermGIndex));
      QuadDebug(cudaFree(constMem.gpuGenPermVarStart));
      QuadDebug(cudaFree(constMem.gpuGenPermVarCount));
      QuadDebug(cudaFree(constMem.cGeneratorCount));
      QuadDebug(cudaDeviceReset());
    }
    
	void InitGPUKernelCuhre(int key, int verbose, int numDevices = 1){
      QuadDebug(cudaDeviceReset());
      KEY = key;
      VERBOSE = verbose;
      NUM_DEVICES = numDevices;
      fEvalPerRegion = (1 + 2*NDIM + 2*NDIM + 2*NDIM + 2*NDIM + 2*NDIM*(NDIM - 1) + 4*NDIM*(NDIM - 1) + 4*NDIM*(NDIM - 1)*(NDIM - 2)/3 + (1 << NDIM));
      QuadDebug(cudaMemcpyToSymbol(dFEvalPerRegion, &fEvalPerRegion, sizeof(size_t), 0, cudaMemcpyHostToDevice));
      Rule.Init(&constMem, fEvalPerRegion, KEY, VERBOSE);
      depthBeingProcessed = 0;
      QuadDebug(Device.SetHeapSize());
    }
	
    size_t getNumActiveRegions(){
      return numRegions;
    }

    void setRegionsData(T *data, size_t size){
      hRegions = &data[0];
      hRegionsLength = &data[size*NDIM];
      numRegions = size;
    }
    
    int GetLastPhase(){return lastPhase;}
    
    size_t Getfregions(){return fregions;}
    
    T *getRegions(size_t size, int startIndex){
      T *newhRegionsAndLength = 0;
      newhRegionsAndLength = (T *)Host.AllocateMemory(&newhRegionsAndLength, 2*sizeof(T)*size*NDIM);
      T *newhRegions = &newhRegionsAndLength[0], *newhRegionsLength = &newhRegionsAndLength[size*NDIM];
      //NOTE:Copy order is important
      for(int dim = 0; dim < NDIM; ++dim){
	QuadDebug(cudaMemcpy(newhRegions + dim * size, dRegions + dim * numRegions + startIndex, sizeof(T) * size, cudaMemcpyDeviceToHost));
	QuadDebug(cudaMemcpy(newhRegionsLength + dim * size, dRegionsLength + dim * numRegions + startIndex, sizeof(T) * size, cudaMemcpyDeviceToHost));
	
      } 
      return newhRegionsAndLength;
    }

    //@brief Template function to display GPU device array variables
    template <class K>
      void display(K *array, size_t size){
      K *tmp = (K *)malloc(sizeof(K) * size);
      cudaMemcpy(tmp, array, sizeof(K)*size, cudaMemcpyDeviceToHost);
      for(int i = 0 ; i < size; ++i){
	printf("%.20lf \n",(T)tmp[i]);
      }
    }

    void
    AllocVolArrays(Volume<T, NDIM>* vol)
    {
      cudaMalloc((void**)&lows, sizeof(T) * NDIM);
      cudaMalloc((void**)&highs, sizeof(T) * NDIM);
      
      
      
      if (vol) {
        cudaMemcpy(lows, vol->lows, sizeof(T) * NDIM, cudaMemcpyHostToDevice);
        cudaMemcpy(highs, vol->highs, sizeof(T) * NDIM, cudaMemcpyHostToDevice);
      } else {
        Volume<T, NDIM> tempVol;
        cudaMemcpy(
          lows, tempVol.lows, sizeof(T) * NDIM, cudaMemcpyHostToDevice);
        cudaMemcpy(
          highs, tempVol.highs, sizeof(T) * NDIM, cudaMemcpyHostToDevice);
      }
    }

    void GenerateInitialRegions(){
      hRegions = (T *)Host.AllocateMemory(&hRegions, sizeof(T)*NDIM);
      hRegionsLength = (T *)Host.AllocateMemory(&hRegionsLength, sizeof(T)*NDIM);
      

      for(int dim = 0 ; dim < NDIM; ++dim){
		hRegions[dim] = 0;
	#if GENZ_TEST == 1
	hRegionsLength[dim] = b[dim];
	#else
	hRegionsLength[dim] = 1;
	#endif
      }
      
      QuadDebug(Device.AllocateMemory((void**)&dRegions, sizeof(T)*NDIM));
      QuadDebug(Device.AllocateMemory((void**)&dRegionsLength, sizeof(T)*NDIM));

      QuadDebug(cudaMemcpy(dRegions, hRegions, sizeof(T) * NDIM, cudaMemcpyHostToDevice));
      QuadDebug(cudaMemcpy(dRegionsLength, hRegionsLength, sizeof(T) * NDIM, cudaMemcpyHostToDevice));

      size_t numThreads = 512;
      size_t numOfDivisionPerRegionPerDimension = 4;
      if(NDIM == 5 )numOfDivisionPerRegionPerDimension = 2;
      if(NDIM == 6 )numOfDivisionPerRegionPerDimension = 2;
      if(NDIM == 7 )numOfDivisionPerRegionPerDimension = 2;
      if(NDIM > 7 )numOfDivisionPerRegionPerDimension = 2;
      if(NDIM > 10 )numOfDivisionPerRegionPerDimension = 1;
      //size_t numOfDivisionPerRegionPerDimension = 1;
      size_t numBlocks = (size_t)ceil(pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM) / numThreads);
      depthBeingProcessed = log2(numOfDivisionPerRegionPerDimension)*NDIM;
      //printf("initial depth being processed:%i\n", depthBeingProcessed);
      numRegions = (size_t)pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM);
  
        
      T *newRegions = 0;
      T *newRegionsLength = 0;
      QuadDebug(Device.AllocateMemory((void **)&newRegions, sizeof(T)*numRegions*NDIM));
      QuadDebug(Device.AllocateMemory((void **)&newRegionsLength, sizeof(T)*numRegions*NDIM));

      generateInitialRegions<T><<<numBlocks, numThreads, NDIM*sizeof(T)>>>(dRegions, dRegionsLength, 1, newRegions, newRegionsLength, numRegions, numOfDivisionPerRegionPerDimension, NDIM);
      
      QuadDebug(Device.ReleaseMemory((void *)dRegions));
      QuadDebug(Device.ReleaseMemory((void *)dRegionsLength));
      
      dRegions = newRegions;
      dRegionsLength = newRegionsLength;
      QuadDebug(cudaMemcpy(dRegions, newRegions, sizeof(T) * numRegions * NDIM, cudaMemcpyDeviceToDevice));
      QuadDebug(cudaMemcpy(dRegionsLength, newRegionsLength, sizeof(T) * numRegions * NDIM, cudaMemcpyDeviceToDevice));

      //QuadDebug(Device.MemoryFree((void **)newRegions));
      //QuadDebug(Device.MemoryFree((void **)newRegionsLength));
     }

    void GenerateActiveIntervals(int *activeRegions, int *subDividingDimension, size_t &nregions){
		
      int *scannedArray = 0;
      QuadDebug(Device.AllocateMemory((void **)&scannedArray, sizeof(int)*numRegions));

      thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(activeRegions);
      thrust::device_ptr<int> scan_ptr = thrust::device_pointer_cast(scannedArray);
      thrust::exclusive_scan(d_ptr, d_ptr + numRegions, scan_ptr);
      
      int last_element;
      size_t numActiveRegions = 0;
     
	  QuadDebug(cudaMemcpy(&last_element, activeRegions + numRegions - 1, sizeof(int), cudaMemcpyDeviceToHost));
      QuadDebug(cudaMemcpy(&numActiveRegions,scannedArray + numRegions - 1, sizeof(int), cudaMemcpyDeviceToHost));
      
	  if(last_element == 1)
		numActiveRegions++;
      
      fregions += numRegions - numActiveRegions;
      if(VERBOSE)
      {
        std::cout << "numActiveRegions\t" << numRegions << "\t" << numActiveRegions << "\n";
      }
      
      nregions += numRegions - numActiveRegions;
      
      if(numActiveRegions > 0){
        depthBeingProcessed++;
		int numOfDivisionOnDimension = 2;
		/*if(numActiveRegions < (1 << 7)){
		  numOfDivisionOnDimension = (1<<11)/numActiveRegions;
		  numOfDivisionOnDimension = 1 << ((int)ceil(log(numOfDivisionOnDimension)/log(2))-1);
		}else if(numActiveRegions < (1<<10)){
		  numOfDivisionOnDimension = 8;
		}else if(numActiveRegions < (1<<12)){
		  numOfDivisionOnDimension = 4;
		}else{
		  numOfDivisionOnDimension = 2;
		  }*/
	
		if(VERBOSE){
		  sprintf(msg, "\nComputing NumOfDivisionsOnDimension\n\t#. of Active Regions\t\t: %ld\n\tDivision on dimension\t\t: %ld division", numActiveRegions, numOfDivisionOnDimension);
		  Println(log, msg);
		}
	
		int *newActiveRegionsBisectDim = 0;
		T *newActiveRegions = 0, *newActiveRegionsLength = 0;
		cudaMalloc((void **)&newActiveRegions, sizeof(T) *  numActiveRegions * NDIM );
		cudaMalloc((void **)&newActiveRegionsLength, sizeof(T) *  numActiveRegions * NDIM);
		cudaMalloc((void **)&newActiveRegionsBisectDim, sizeof(int) * numActiveRegions * numOfDivisionOnDimension);

		size_t numThreads = BLOCK_SIZE;
		size_t numBlocks = numRegions/numThreads + ((numRegions%numThreads)?1:0);
		
		if(VERBOSE){
		  Println(log, "\nCalling GPU Function align_intervals");
		  sprintf(msg, "\n\t# of input intervals\t\t: %ld\n\t#. of Active Intervals\t\t: %ld\n\t#. of Thread Blocks\t\t: %ld\n\t#. of Threads per Blocks\t: %ld\n",numRegions, numActiveRegions, numBlocks, numThreads);
		  Println(log, msg);
		}

		alignRegions<T, NDIM><<<numBlocks, numThreads>>>(dRegions, dRegionsLength, activeRegions, subDividingDimension, scannedArray, newActiveRegions, newActiveRegionsLength, newActiveRegionsBisectDim, numRegions, numActiveRegions, numOfDivisionOnDimension);
	
		if(VERBOSE){
		  Println(log, "\nCalling GPU Function divideIntervalsGPU");
		  sprintf(msg, "\n\t# of input intervals\t\t: %ld\n\t#. of division on dimension\t: %ld\n\t#. of Thread Blocks\t\t: %ld\n\t#. of Threads per Blocks\t: %ld",numActiveRegions, numOfDivisionOnDimension, numBlocks, numThreads);
		  Println(log, msg);
		}
	
		T *genRegions = 0, *genRegionsLength = 0;
		numBlocks = numActiveRegions/numThreads + ((numActiveRegions%numThreads)?1:0);
        
		QuadDebug(cudaMalloc((void **)&genRegions, sizeof(T) * numActiveRegions * NDIM * numOfDivisionOnDimension));
		QuadDebug(cudaMalloc((void **)&genRegionsLength, sizeof(T) * numActiveRegions * NDIM * numOfDivisionOnDimension));
		
		divideIntervalsGPU<T, NDIM><<<numBlocks, numThreads>>>(genRegions, genRegionsLength, newActiveRegions, newActiveRegionsLength, newActiveRegionsBisectDim, numActiveRegions, numOfDivisionOnDimension);
		QuadDebug(Device.ReleaseMemory(newActiveRegions));
		QuadDebug(Device.ReleaseMemory(newActiveRegionsLength));
		QuadDebug(Device.ReleaseMemory(newActiveRegionsBisectDim));
		
		numRegions = numActiveRegions * numOfDivisionOnDimension;
		QuadDebug(Device.ReleaseMemory((void *)dRegions));
		QuadDebug(Device.ReleaseMemory((void *)dRegionsLength));
		QuadDebug(Device.ReleaseMemory((void *)scannedArray));
		
		dRegions = genRegions;
		dRegionsLength = genRegionsLength;
		//TODO: throws error
		//QuadDebug(cudaMemcpy(dRegions, genRegions, sizeof(T) * numRegions * NDIM, cudaMemcpyDeviceToDevice));
		//QuadDebug(cudaMemcpy(dRegionsLength, genRegionsLength, sizeof(T) * numRegions * NDIM, cudaMemcpyDeviceToDevice));
		}
		else{
			numRegions = 0;
		}
    }

    template <typename IntegT>
    void FirstPhaseIteration(IntegT* d_integrand, T epsrel, T epsabs, T &integral, T &error, size_t &nregions, size_t &neval){
      /*
      std::stringstream ss, *gss;
      ss << std::setprecision(16);
      ss << std::scientific;
      int num_gpus = -1;
      /////////////////////////////////////////////////////////////////
      // determine the number of CUDA capable GPUs
      //
      cudaGetDeviceCount(&num_gpus);
      if(num_gpus < 1){
	fprintf(stderr,"no CUDA capable devices were detected\n");
	exit(1);
      }
      int num_cpu_procs = omp_get_num_procs();
      if(VERBOSE){
	/////////////////////////////////////////////////////////////////
	// display CPU and GPU configuration
	sprintf(msg, "number of host CPUs:\t%d\n", omp_get_num_procs());
	Println(log, msg);
	sprintf(msg,"number of CUDA devices:\t%d\n", num_gpus);
	Println(log, msg);
	for(int i = 0; i < num_gpus; i++){
	  cudaDeviceProp dprop;
	  cudaGetDeviceProperties(&dprop, i);
	  sprintf(msg,"   %d: %s\n", i, dprop.name);
	  Println(log, msg);
	}
	Println(log, "---------------------------\n");
      }

      if(NUM_DEVICES > num_gpus)
	NUM_DEVICES = num_gpus;
      omp_set_num_threads(NUM_DEVICES);
      cudaStream_t stream[NUM_DEVICES];
      cudaEvent_t event[NUM_DEVICES];

#pragma omp parallel
      {	
	unsigned int cpu_thread_id = omp_get_thread_num();
	unsigned int num_cpu_threads = omp_get_num_threads();
		
	// set and check the CUDA device for this CPU thread
	int gpu_id = -1;
	
	QuadDebug(cudaSetDevice(cpu_thread_id % num_gpus));	// "% num_gpus" allows more CPU threads than GPU devices
	QuadDebug(cudaGetDevice(&gpu_id));
	warmUpKernel<<<FIRST_PHASE_MAXREGIONS, BLOCK_SIZE>>>();


	if(VERBOSE){
	  sprintf(msg, "CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
	  Println(log, msg);
	}
	
	if(cpu_thread_id < num_cpu_threads){	
	  size_t numRegionsThread = numRegions/num_cpu_threads;
	  int startIndex = cpu_thread_id * numRegionsThread;
	  int endIndex = (cpu_thread_id+1) * numRegionsThread;
	  if(cpu_thread_id == (num_cpu_threads-1)){
	    endIndex = numRegions;
	  }
	  numRegionsThread = endIndex - startIndex;

	  
	  QuadDebug(Device.SetHeapSize());
	  Rule.loadDeviceConstantMemory(cpu_thread_id);

	  size_t numThreads = BLOCK_SIZE;
	  size_t numBlocks = numRegionsThread;
	  T *dRegionsError = 0, *dRegionsIntegral = 0;
	  T *dRegionsThread = 0, *dRegionsLengthThread = 0;
	  QuadDebug(Device.AllocateMemory((void **)&dRegionsIntegral, sizeof(T)*numRegionsThread*2));
	  QuadDebug(Device.AllocateMemory((void **)&dRegionsError, sizeof(T)*numRegionsThread*2));
	 
	  int *activeRegions = 0, *subDividingDimension = 0;
	  QuadDebug(Device.AllocateMemory((void **)&activeRegions, sizeof(int)*numRegionsThread));
	  QuadDebug(Device.AllocateMemory((void **)&subDividingDimension, sizeof(int)*numRegionsThread));
 
	}
      }
      exit(1);
      */
      size_t numThreads = BLOCK_SIZE;
      size_t numBlocks = numRegions;
      T *dRegionsError = 0, *dRegionsIntegral = 0;
      QuadDebug(Device.AllocateMemory((void **)&dRegionsIntegral, sizeof(T)*numRegions*2));
      QuadDebug(Device.AllocateMemory((void **)&dRegionsError, sizeof(T)*numRegions*2));

      int *activeRegions = 0, *subDividingDimension = 0;
      QuadDebug(Device.AllocateMemory((void **)&activeRegions, sizeof(int)*numRegions));      
      QuadDebug(Device.AllocateMemory((void **)&subDividingDimension, sizeof(int)*numRegions));

      
      if(VERBOSE){
		Println(log, "\nEntering function IntegrateFirstPhase \n");
		sprintf(msg, "\t# of input intervals\t\t: %ld\n\t#. of Thread Blocks\t\t: %ld\n\t#. of Threads per Blocks\t: %ld\n",numRegions, numBlocks, numThreads);
		Println(log, msg);
      }

      INTEGRATE_GPU_PHASE1<IntegT, T, NDIM><<<numBlocks, numThreads, NDIM * sizeof(GlobalBounds)>>>(d_integrand, dRegions, dRegionsLength, numRegions, dRegionsIntegral, dRegionsError, activeRegions, subDividingDimension, epsrel, epsabs, constMem,  Rule.GET_FEVAL(), Rule.GET_NSETS(), lows, highs, depthBeingProcessed);

      //nregions += numRegions;
      neval += numRegions*fEvalPerRegion;
    
      thrust::device_ptr<T> wrapped_ptr;
      wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral + numRegions);
      T rG = integral + thrust::reduce(wrapped_ptr, wrapped_ptr+numRegions);

      wrapped_ptr = thrust::device_pointer_cast(dRegionsError + numRegions);
      T errG = error + thrust::reduce(wrapped_ptr, wrapped_ptr+numRegions);
      
      wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral);
      integral = integral + thrust::reduce(wrapped_ptr, wrapped_ptr+numRegions);

      wrapped_ptr = thrust::device_pointer_cast(dRegionsError);
      error = error + thrust::reduce(wrapped_ptr, wrapped_ptr+numRegions);
      
      //std::cout << "Error " << errG << " " << rG<< std::endl;
     
      if((errG <= MaxErr(rG, epsrel, epsabs)) && GLOBAL_ERROR) {
		nregions += numRegions;
		if(VERBOSE){
		  sprintf(msg, "Global Error Check -\t%ld integrand evaluations so far\n%lf +- %lf ", neval, rG, errG);
		  Println(log, msg);
		}
		integral = rG;
		error = errG;
		numRegions = 0;
		return;
      }
	  
      GenerateActiveIntervals(activeRegions, subDividingDimension, nregions);
      
      QuadDebug(cudaFree(subDividingDimension));
      QuadDebug(cudaFree(activeRegions));
      QuadDebug(cudaFree(dRegionsError));
      QuadDebug(cudaFree(dRegionsIntegral));      
    }

    template <typename IntegT>
    void IntegrateFirstPhase(IntegT* d_integrand, T epsrel, T epsabs, T &integral, T &error, size_t &nregions, size_t &neval, Volume<double, NDIM>* vol = nullptr){
      AllocVolArrays(vol);
      for(int i  = 0; i < 100; i++){
		FirstPhaseIteration<IntegT>(d_integrand, epsrel, epsabs, integral, error, nregions, neval);
		//printf("iteration %i %.15e +- %.15e\n", i, integral, error);
		if(VERBOSE){
		  sprintf(msg, "Iterations %d:\t%ld integrand evaluations so far\n%lf +- %lf ", i+1 , neval, integral, error);
		  Println(log, msg);
		  sprintf(msg, "\n==========================================================================\n");
		  Println(log, msg);
		}	
	
		if(numRegions < 1 && nregions > 1) 
			return;
        //2*numRegions > FIRST_PHASE_MAXREGIONS 
        //2*numRegions > FIRST_PHASE_MAXREGIONS
		if(numRegions > FIRST_PHASE_MAXREGIONS/*2*numRegions > FIRST_PHASE_MAXREGIONS*/) //to have max FIRST_PHASE_MAXREGIONS blocks instead FIRST_PHASE_MAXREGIONS*2 
			break;
        
      }
	  
      hRegions = (T *)Host.AllocateMemory(&hRegions, sizeof(T) * numRegions * NDIM);
      hRegionsLength = (T *)Host.AllocateMemory(&hRegionsLength, sizeof(T) * numRegions * NDIM);
      QuadDebug(cudaMemcpy(hRegions, dRegions, sizeof(T) * numRegions * NDIM, cudaMemcpyDeviceToHost));
      QuadDebug(cudaMemcpy(hRegionsLength, dRegionsLength, sizeof(T) * numRegions * NDIM, cudaMemcpyDeviceToHost));
      
    }

    template <typename IntegT>
    int IntegrateSecondPhase(IntegT* d_integrand, T epsrel, T epsabs, T &integral, T &error, size_t &nregions, size_t &neval, T *optionalInfo = 0){
	  int numFailedRegions = 0;
      int num_gpus = 0;	// number of CUDA GPUs
	  
      if(optionalInfo!=0){
		optionalInfo[0] = -INFTY;
      }
      
      /////////////////////////////////////////////////////////////////
      // determine the number of CUDA capable GPUs
      //
      cudaGetDeviceCount(&num_gpus);
      if(num_gpus < 1){
		fprintf(stderr, "no CUDA capable devices were detected\n");
		exit(1);
      }
      int num_cpu_procs = omp_get_num_procs();

      /*
	Why did you have this section?
      for(int i = 1; i < num_gpus; i++){
	int gpu_id;
	QuadDebug(cudaSetDevice(i));	// "% num_gpus" allows more CPU threads than GPU devices
	QuadDebug(cudaGetDevice(&gpu_id));
	QuadDebug(cudaDeviceReset());
      }	  
      */

      if(VERBOSE){
		/////////////////////////////////////////////////////////////////
		// display CPU and GPU configuration
		sprintf(msg, "number of host CPUs:\t%d\n", omp_get_num_procs());
		printf("number of host CPUs:\t%d\n", omp_get_num_procs());
		Println(log, msg);
		sprintf(msg,"number of CUDA devices:\t%d\n", num_gpus);
		printf("number of CUDA devices:\t%d\n", num_gpus);
		Println(log, msg);
		for(int i = 0; i < num_gpus; i++){
		  cudaDeviceProp dprop;
		  cudaGetDeviceProperties(&dprop, i);
		  sprintf(msg,"   %d: %s\n", i, dprop.name);
		  Println(log, msg);
		}
		Println(log, "---------------------------\n");
      }
	
	//this works ok, check command line arg with actual devices available
      if(NUM_DEVICES > num_gpus)
		NUM_DEVICES = num_gpus;
	  
      omp_set_num_threads(NUM_DEVICES);
      cudaStream_t stream[NUM_DEVICES];
      cudaEvent_t event[NUM_DEVICES];
		
#pragma omp parallel

      {	
		unsigned int cpu_thread_id = omp_get_thread_num();
		unsigned int num_cpu_threads = omp_get_num_threads();
	
		// set and check the CUDA device for this CPU thread
		int gpu_id = -1;
	   
		QuadDebug(cudaSetDevice(cpu_thread_id % num_gpus));	// "% num_gpus" allows more CPU threads than GPU devices
		QuadDebug(cudaGetDevice(&gpu_id));
		warmUpKernel<<<FIRST_PHASE_MAXREGIONS, BLOCK_SIZE>>>();
		
		if(VERBOSE){
		  sprintf(msg, "CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
		  Println(log, msg);
		}
	
		if(cpu_thread_id < num_cpu_threads){
		
			size_t numRegionsThread = numRegions/num_cpu_threads;
			int startIndex = cpu_thread_id * numRegionsThread;
			int endIndex = (cpu_thread_id+1) * numRegionsThread;
			if(cpu_thread_id == (num_cpu_threads-1))
				endIndex = numRegions;
			  
			numRegionsThread = endIndex - startIndex;
			  
			if(VERBOSE){
				printf("Num Regions in Phase 2:%i (%i)\n", numRegions, cpu_thread_id);
				printf("Number of Regions going to each GPU:%d (%i)\n", numRegionsThread, cpu_thread_id);
				printf("startIndex:%i (%i)\n", startIndex, cpu_thread_id);
				printf("endIndex:%i (%i)\n", endIndex, cpu_thread_id);
			} 
	  //QuadDebug(Device.SetHeapSize());
	  CudaCheckError();
	
	  Rule.loadDeviceConstantMemory(&constMem, cpu_thread_id);
	  size_t numThreads = BLOCK_SIZE;
	  size_t numBlocks = numRegionsThread;
	  
	  T *dRegionsError = 0, *dRegionsIntegral = 0;
	  T *dRegionsThread = 0, *dRegionsLengthThread = 0;
	  
	  QuadDebug(Device.AllocateMemory((void **)&dRegionsIntegral, sizeof(T)*numRegionsThread));
	  QuadDebug(Device.AllocateMemory((void **)&dRegionsError, sizeof(T)*numRegionsThread));
      
	  int *activeRegions = 0, *subDividingDimension = 0, *dRegionsNumRegion = 0;
	  
	  QuadDebug(Device.AllocateMemory((void **)&activeRegions, sizeof(int)*numRegionsThread));      
	  QuadDebug(Device.AllocateMemory((void **)&subDividingDimension, sizeof(int)*numRegionsThread));
	  QuadDebug(Device.AllocateMemory((void **)&dRegionsNumRegion, sizeof(int)*numRegionsThread));      
	  QuadDebug(Device.AllocateMemory((void **)&dRegionsThread, sizeof(T)*numRegionsThread * NDIM));
	  QuadDebug(Device.AllocateMemory((void **)&dRegionsLengthThread, sizeof(T)*numRegionsThread * NDIM));
	
	  CudaCheckError();
	  //NOTE:Copy order is important
	  
	  for(int dim = 0; dim < NDIM; ++dim){
		/*printf("copying from hRegions(%i-%i) -> dRegions(%i-%i) (%i) |numRegionsThread:%i numRegions:%i\n", dim * numRegions + startIndex , dim * numRegions + startIndex+numRegionsThread,
																					dim * numRegionsThread, dim * numRegionsThread+numRegionsThread,
																					cpu_thread_id, numRegionsThread, numRegions);*/
	    QuadDebug(cudaMemcpy(dRegionsThread + dim * numRegionsThread, 
							 hRegions 		+ dim * numRegions + startIndex, 
							 sizeof(T) * numRegionsThread, 
							 cudaMemcpyHostToDevice));
	    
		QuadDebug(cudaMemcpy(dRegionsLengthThread + dim * numRegionsThread, 
							 hRegionsLength + dim * numRegions + startIndex, 
							 sizeof(T) * numRegionsThread, 
							 cudaMemcpyHostToDevice));
	  }
	  
	  CudaCheckError();
	  
	  cudaEvent_t start;
	  QuadDebug(cudaStreamCreate(&stream[gpu_id]));
	  QuadDebug(cudaEventCreate(&start));
	  QuadDebug(cudaEventCreate(&event[gpu_id]));
	  QuadDebug(cudaEventRecord(start, stream[gpu_id]));
	  CudaCheckError();
	  
	  if(VERBOSE){
	    Println(log, "\n GPU Function PHASE2");
	    sprintf(msg, "\t# of input intervals\t\t: %ld\n\t#. of Thread Blocks\t\t: %ld\n\t#. of Threads per Blocks\t: %ld\n",numRegionsThread, numBlocks, numThreads);
        std::cout << " phase2 : 	blocks:" << numBlocks << " threads:" << numThreads << std::endl; 
		//printf(msg, "\t# of input intervals\t\t: %ld\n\t#. of Thread Blocks\t\t: %ld\n\t#. of Threads per Blocks\t: %ld\n",numRegionsThread, numBlocks, numThreads);
	    Println(log, msg);
	  }
  	  CudaCheckError();
    
	 cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeof(Region<NDIM>)*numBlocks*MAX_GLOBALPOOL_SIZE);
     CudaCheckError();
     printf("launching with %lu blocks\n", numBlocks);
	 BLOCK_INTEGRATE_GPU_PHASE2<IntegT, T, NDIM><<<numBlocks, numThreads, NDIM * sizeof(GlobalBounds), stream[gpu_id]>>>(d_integrand, dRegionsThread, dRegionsLengthThread, numRegionsThread, dRegionsIntegral, dRegionsError, dRegionsNumRegion, activeRegions, subDividingDimension, epsrel, epsabs, gpu_id, constMem, Rule.GET_FEVAL(), Rule.GET_NSETS(), lows, highs, depthBeingProcessed);
	  
	  cudaDeviceSynchronize();
	  //printf("BLOCK INTEGRATE_GPU done %d gpu:%i\n", cpu_thread_id, gpu_id);
	  CudaCheckError();
	  //printf("After error checking and sync %d\n", cpu_thread_id);
	  cudaDeviceSynchronize();
	  cudaEventRecord( event[gpu_id], stream[gpu_id]);
	  cudaEventSynchronize( event[gpu_id] );
	 
	  float elapsed_time;
	  cudaEventElapsedTime(&elapsed_time, start, event[gpu_id]);
	  if(optionalInfo!=0 && elapsed_time > optionalInfo[0]){
	    optionalInfo[0] = elapsed_time;
	  }
		
	  //if(VERBOSE){
	  //  sprintf(msg, "\nSecond Phase Kernel by thread %d (of %d) using CUDA device %d took %.1f ms ", cpu_thread_id, num_cpu_threads, gpu_id, elapsed_time);
	 //   Println(log, msg);
	  //}
	  
	  //cudaEventDestroy(start);
	  //cudaEventDestroy(event[gpu_id]);

	  thrust::device_ptr<T> wrapped_ptr;
	  wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral);
	  T integResult = thrust::reduce(wrapped_ptr, wrapped_ptr+numRegionsThread);
	  
	  integral += integResult; 

	  
	  wrapped_ptr = thrust::device_pointer_cast(dRegionsError);
	  error = error + thrust::reduce(wrapped_ptr, wrapped_ptr+numRegionsThread);
      
	  thrust::device_ptr<int> int_ptr = thrust::device_pointer_cast(dRegionsNumRegion);
	  int regionCnt = thrust::reduce(int_ptr, int_ptr+numRegionsThread);
	  nregions += regionCnt;
	  //std::cout << "Num regions : " << regionCnt << std::endl;
      
	  neval += (regionCnt - numRegionsThread)*fEvalPerRegion*2+numRegionsThread*fEvalPerRegion;
 
	  int_ptr = thrust::device_pointer_cast(activeRegions);
	  numFailedRegions += thrust::reduce(int_ptr, int_ptr+numRegionsThread);

	  //std::cout << "--" << numFailedRegions << std::endl;
	  //QuadDebug(cudaThreadExit());
	  
	  QuadDebug(Device.ReleaseMemory(dRegionsError));
	  QuadDebug(Device.ReleaseMemory(dRegionsIntegral));
	  QuadDebug(Device.ReleaseMemory(dRegionsThread));
	  QuadDebug(Device.ReleaseMemory(dRegionsLengthThread));
	  QuadDebug(Device.ReleaseMemory(activeRegions));
	  QuadDebug(Device.ReleaseMemory(subDividingDimension));
	  QuadDebug(Device.ReleaseMemory(dRegionsNumRegion));
	  QuadDebug(cudaDeviceSynchronize());
	}
	else
		printf("Rogue cpu thread\n");
  }
      lastPhase = 2;
      //sprintf(msg, "Execution time : %.2lf", optionalInfo[0]);
      //Print(msg);
      return numFailedRegions;

    }

  };
   
}
