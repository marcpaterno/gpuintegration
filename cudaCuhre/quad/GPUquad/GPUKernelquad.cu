#ifndef CUDACUHRE_QUAD_GPUQUAD_GPUKERNELQUAD_CUH
#define CUDACUHRE_QUAD_GPUQUAD_GPUKERNELQUAD_CUH

#include "GPUQuadPhases.cu"
#include "GPUQuadRule.cu"

namespace quad {
  using namespace cooperative_groups;

  //===========
  // FOR DEBUGGINGG

  bool
  cudaMemoryTest()
  {
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int* h_a = (int*)malloc(bytes);
    int* d_a;
    cudaMalloc((int**)&d_a, bytes);

    memset(h_a, 0, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    return true;
  }

  //==========

  __constant__ size_t dFEvalPerRegion;

  template <typename T>
  __global__ void
  PrintcuArray(T* array, int size)
  {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int i = 0; i < size; i++) {
        // if(i<10)
        printf("array[%i]:%.12f\n", i, array[i]);
        printf("array[%i]:%.12f\n", i, array[i]);
      }
    }
  }

  template <typename T>
  __global__ void
  PrintcuArray(T* array, T* array2, int size)
  {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int i = 0; i < size; i++)
        // if(i<10)
        printf("array[%i]:%.12f - %.12f\n", i, array[i], array[i] + array2[i]);
    }
  }

  template <typename T>
  __global__ void
  generateInitialRegions(T* dRegions,
                         T* dRegionsLength,
                         size_t numRegions,
                         T* newRegions,
                         T* newRegionsLength,
                         size_t newNumOfRegions,
                         int numOfDivisionsPerRegionPerDimension,
                         int NDIM)
  {

    extern __shared__ T slength[];
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;


    if (threadIdx.x < NDIM) {
      slength[threadIdx.x] =
        dRegionsLength[threadIdx.x] / numOfDivisionsPerRegionPerDimension;
    }
    __syncthreads();

    if (threadId < newNumOfRegions) {
      size_t interval_index =
        threadId / pow((T)numOfDivisionsPerRegionPerDimension, (T)NDIM);
      size_t local_id =
        threadId % (size_t)pow((T)numOfDivisionsPerRegionPerDimension, (T)NDIM);
      for (int dim = 0; dim < NDIM; ++dim) {
        size_t id =
          (size_t)(local_id /
                   pow((T)numOfDivisionsPerRegionPerDimension, (T)dim)) %
          numOfDivisionsPerRegionPerDimension;
        newRegions[newNumOfRegions * dim + threadId] =
          dRegions[numRegions * dim + interval_index] + id * slength[dim];
        newRegionsLength[newNumOfRegions * dim + threadId] = slength[dim];
      }
    }
  }

  template <typename T>
  __global__ void
  alignRegions(T* dRegions,
                T* dRegionsLength,
                int* activeRegions,
                T* dRegionsIntegral,
                T* dRegionsError,
                T* dRegionsParentIntegral,
                T* dRegionsParentError,
                int* subDividingDimension,
                int* scannedArray,
                T* newActiveRegions,
                T* newActiveRegionsLength,
                int* newActiveRegionsBisectDim,
                size_t numRegions,
                size_t newNumRegions,
                int numOfDivisionOnDimension)
  {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numRegions && activeRegions[tid] == 1) {
      size_t interval_index = scannedArray[tid];

      for (int i = 0; i < DIM; ++i) {
        newActiveRegions[i * newNumRegions + interval_index] =
          dRegions[i * numRegions + tid];
        newActiveRegionsLength[i * newNumRegions + interval_index] =
          dRegionsLength[i * numRegions + tid];
      }

      dRegionsParentIntegral[interval_index] =
        dRegionsIntegral[tid + numRegions];
      dRegionsParentError[interval_index] = dRegionsError[tid + numRegions];

      dRegionsParentIntegral[interval_index + newNumRegions] =
        dRegionsIntegral[tid + numRegions];
      dRegionsParentError[interval_index + newNumRegions] =
        dRegionsError[tid + numRegions];

      for (int i = 0; i < numOfDivisionOnDimension; ++i) {
        newActiveRegionsBisectDim[i * newNumRegions + interval_index] =
          subDividingDimension[tid];
      }
    }
  }

  template <typename T>
  __global__ void
  divideIntervalsGPU(T* genRegions,
                     T* genRegionsLength,
                     T* activeRegions,
                     T* activeRegionsLength,
                     int* activeRegionsBisectDim,
                     size_t numActiveRegions,
                     int numOfDivisionOnDimension)
  {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numActiveRegions) {

      int bisectdim = activeRegionsBisectDim[tid];
      size_t data_size = numActiveRegions * numOfDivisionOnDimension;

      for (int i = 0; i < numOfDivisionOnDimension; ++i) {
        for (int dim = 0; dim < DIM; ++dim) {
          genRegions[i * numActiveRegions + dim * data_size + tid] =
            activeRegions[dim * numActiveRegions + tid];
          genRegionsLength[i * numActiveRegions + dim * data_size + tid] =
            activeRegionsLength[dim * numActiveRegions + tid];    
        }
      }

      for (int i = 0; i < numOfDivisionOnDimension; ++i) {

        T interval_length =
          activeRegionsLength[bisectdim * numActiveRegions + tid] /
          numOfDivisionOnDimension;
        genRegions[bisectdim * data_size + i * numActiveRegions + tid] =
          activeRegions[bisectdim * numActiveRegions + tid] +
          i * interval_length;
        genRegionsLength[i * numActiveRegions + bisectdim * data_size + tid] =
          interval_length;
      }
    }
  }

  template <typename T>
  class GPUKernelCuhre {
    T* dRegions;
    T* dRegionsLength;
    T* hRegions;
    T* hRegionsLength;
    int NDIM, KEY, VERBOSE;
    size_t numRegions, numFunctionEvaluations;
    size_t fEvalPerRegion;
    HostMemory<T> Host;
    DeviceMemory<T> Device;
    QuadRule<T> Rule;
    Structures<T> constMem;
    int NUM_DEVICES;
    // Debug Msg
    char msg[256];

    std::ostream& log;

  public:
    void
    ExpandcuArray(T*& array, int currentSize, int newSize)
    {
      T* temp = 0;
      QuadDebug(Device.AllocateMemory((void**)&temp, sizeof(T) * newSize));
      QuadDebug(cudaMemcpy(
        temp, array, sizeof(T) * currentSize, cudaMemcpyDeviceToDevice));
      QuadDebug(Device.ReleaseMemory(array));
      array = temp;
    }

    GPUKernelCuhre(std::ostream& logerr = std::cout) : log(logerr)
    {
      numRegions = 0;
      numFunctionEvaluations = 0;
      NDIM = 0;
      KEY = 0;
    }

    ~GPUKernelCuhre()
    {

      if (VERBOSE) {
        sprintf(msg, "GPUKerneCuhre Destructur");
        Println(log, msg);
      }

      QuadDebug(Device.ReleaseMemory(dRegions));
      QuadDebug(Device.ReleaseMemory(dRegionsLength));
      Host.ReleaseMemory(hRegions);
      Host.ReleaseMemory(hRegionsLength);
      QuadDebug(cudaDeviceReset());
      // commented out by Ioannis, needs to be addressed
      // if(DIM > 8)
      // QuadDebug(Device.ReleaseMemory(gpuGenPos));
    }

    size_t
    getNumActiveRegions()
    {
      return numRegions;
    }

    void
    setRegionsData(T* data, size_t size)
    {
      hRegions = &data[0];
      hRegionsLength = &data[size * NDIM];
      numRegions = size;
    }

    T*
    getRegions(size_t size, int startIndex)
    {
      T* newhRegionsAndLength = 0;
      newhRegionsAndLength = (T*)Host.AllocateMemory(
        &newhRegionsAndLength, 2 * sizeof(T) * size * NDIM);
      T *newhRegions = &newhRegionsAndLength[0],
        *newhRegionsLength = &newhRegionsAndLength[size * NDIM];
      // NOTE:Copy order is important
      for (int dim = 0; dim < NDIM; ++dim) {
        QuadDebug(cudaMemcpy(newhRegions + dim * size,
                             dRegions + dim * numRegions + startIndex,
                             sizeof(T) * size,
                             cudaMemcpyDeviceToHost));
        QuadDebug(cudaMemcpy(newhRegionsLength + dim * size,
                             dRegionsLength + dim * numRegions + startIndex,
                             sizeof(T) * size,
                             cudaMemcpyDeviceToHost));
      }
      return newhRegionsAndLength;
    }

    void
    InitGPUKernelCuhre(int dim, int key, int verbose, int numDevices = 1)
    {
      QuadDebug(cudaDeviceReset());
      NDIM = dim;
      KEY = key;
      VERBOSE = verbose;
      NUM_DEVICES = numDevices;
      fEvalPerRegion = (1 + 2 * NDIM + 2 * NDIM + 2 * NDIM + 2 * NDIM +
                        2 * NDIM * (NDIM - 1) + 4 * NDIM * (NDIM - 1) +
                        4 * NDIM * (NDIM - 1) * (NDIM - 2) / 3 + (1 << NDIM));
      QuadDebug(cudaMemcpyToSymbol(dFEvalPerRegion,
                                   &fEvalPerRegion,
                                   sizeof(size_t),
                                   0,
                                   cudaMemcpyHostToDevice));
      Rule.Init(NDIM, fEvalPerRegion, KEY, VERBOSE, &constMem);
      QuadDebug(Device.SetHeapSize());
    }

    //@brief Template function to display GPU device array variables
    template <class K>
    void
    display(K* array, size_t size)
    {
      K* tmp = (K*)malloc(sizeof(K) * size);
      cudaMemcpy(tmp, array, sizeof(K) * size, cudaMemcpyDeviceToHost);
      for (int i = 0; i < size; ++i) {
        printf("%.20lf \n", (T)tmp[i]);
      }
    }

    void
    GenerateInitialRegions()
    {
      hRegions = (T*)Host.AllocateMemory(&hRegions, sizeof(T) * NDIM);
      hRegionsLength =
        (T*)Host.AllocateMemory(&hRegionsLength, sizeof(T) * NDIM);

      for (int dim = 0; dim < NDIM; ++dim) {
        hRegions[dim] = 0;
#if GENZ_TEST == 1
        hRegionsLength[dim] = b[dim];
#else
        hRegionsLength[dim] = 1;
#endif
      }

      QuadDebug(Device.AllocateMemory((void**)&dRegions, sizeof(T) * NDIM));
      QuadDebug(
        Device.AllocateMemory((void**)&dRegionsLength, sizeof(T) * NDIM));

      QuadDebug(cudaMemcpy(
        dRegions, hRegions, sizeof(T) * NDIM, cudaMemcpyHostToDevice));
      QuadDebug(cudaMemcpy(dRegionsLength,
                           hRegionsLength,
                           sizeof(T) * NDIM,
                           cudaMemcpyHostToDevice));

      size_t numThreads = 512;
	  //this has been changed temporarily, do not remove
      /*size_t numOfDivisionPerRegionPerDimension = 4;
      if(NDIM == 5 )numOfDivisionPerRegionPerDimension = 2;
      if(NDIM == 6 )numOfDivisionPerRegionPerDimension = 2;
      if(NDIM == 7 )numOfDivisionPerRegionPerDimension = 2;
      if(NDIM > 7 )numOfDivisionPerRegionPerDimension = 2;
      if(NDIM > 10 )numOfDivisionPerRegionPerDimension = 1;*/

      size_t numOfDivisionPerRegionPerDimension = 1;

      size_t numBlocks = (size_t)ceil(
        pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM) / numThreads);
      numRegions = (size_t)pow((T)numOfDivisionPerRegionPerDimension, (T)NDIM);

      T* newRegions = 0;
      T* newRegionsLength = 0;
      QuadDebug(Device.AllocateMemory((void**)&newRegions,
                                      sizeof(T) * numRegions * NDIM));
      QuadDebug(Device.AllocateMemory((void**)&newRegionsLength,
                                      sizeof(T) * numRegions * NDIM));

      generateInitialRegions<T><<<numBlocks, numThreads, NDIM * sizeof(T)>>>(
        dRegions,
        dRegionsLength,
        1,
        newRegions,
        newRegionsLength,
        numRegions,
        numOfDivisionPerRegionPerDimension,
        NDIM);

      QuadDebug(Device.ReleaseMemory((void*)dRegions));
      QuadDebug(Device.ReleaseMemory((void*)dRegionsLength));

      dRegions = newRegions;
      dRegionsLength = newRegionsLength;
      QuadDebug(cudaMemcpy(dRegions,
                           newRegions,
                           sizeof(T) * numRegions * NDIM,
                           cudaMemcpyDeviceToDevice));
      QuadDebug(cudaMemcpy(dRegionsLength,
                           newRegionsLength,
                           sizeof(T) * numRegions * NDIM,
                           cudaMemcpyDeviceToDevice));
    }

    void
    GenerateActiveIntervals(int* activeRegions,
                             int* subDividingDimension,
                             T* dRegionsIntegral,
                             T* dRegionsError,
                             T*& dParentsIntegral,
                             T*& dParentsError)
    {

      int* scannedArray = 0;
      QuadDebug(
        Device.AllocateMemory((void**)&scannedArray, sizeof(int) * numRegions));

      thrust::device_ptr<int> d_ptr =
        thrust::device_pointer_cast(activeRegions);
      thrust::device_ptr<int> scan_ptr =
        thrust::device_pointer_cast(scannedArray);
      thrust::exclusive_scan(d_ptr, d_ptr + numRegions, scan_ptr);

      int last_element;
      size_t numActiveRegions = 0;

      QuadDebug(cudaMemcpy(&last_element,
                           activeRegions + numRegions - 1,
                           sizeof(int),
                           cudaMemcpyDeviceToHost));
      QuadDebug(cudaMemcpy(&numActiveRegions,
                           scannedArray + numRegions - 1,
                           sizeof(int),
                           cudaMemcpyDeviceToHost));

      if (last_element == 1)
        numActiveRegions++;

      if (VERBOSE) {
        printf("numRegions:%lu \t numActiveRegions (Bad Regions):%lu\n",
               numRegions,
               numActiveRegions);
      }

      if (numActiveRegions > 0) {

        int numOfDivisionOnDimension = 2;

        if (VERBOSE) {
          sprintf(msg,
                  "\nComputing NumOfDivisionsOnDimension\n\t#. of Active "
                  "Regions\t\t: %lu\n\tDivision on dimension\t\t: %i division",
                  numActiveRegions,
                  numOfDivisionOnDimension);
          Println(log, msg);
        }

        int* newActiveRegionsBisectDim = 0;
        T *newActiveRegions = 0, *newActiveRegionsLength = 0;

        cudaMalloc((void**)&newActiveRegions,
                   sizeof(T) * numActiveRegions * NDIM);
        cudaMalloc((void**)&newActiveRegionsLength,
                   sizeof(T) * numActiveRegions * NDIM);

        ExpandcuArray(dParentsIntegral, numRegions * 2, numActiveRegions * 4);
        ExpandcuArray(dParentsError, numRegions * 2, numActiveRegions * 4);

        cudaMalloc((void**)&newActiveRegionsBisectDim,
                   sizeof(int) * numActiveRegions * numOfDivisionOnDimension);

        size_t numThreads = BLOCK_SIZE;
        size_t numBlocks =
          numRegions / numThreads + ((numRegions % numThreads) ? 1 : 0);

        if (VERBOSE) {
          Println(log, "\nCalling GPU Function align_intervals");
          sprintf(msg,
                  "\n\t# of input intervals\t\t: %ld\n\t#. of Active "
                  "Intervals\t\t: %ld\n\t#. of Thread Blocks\t\t: %ld\n\t#. of "
                  "Threads per Blocks\t: %ld\n",
                  numRegions,
                  numActiveRegions,
                  numBlocks,
                  numThreads);

          Println(log, msg);
        }

        cudaDeviceSynchronize();

        alignRegions<T><<<numBlocks, numThreads>>>(dRegions,
                                                    dRegionsLength,
                                                    activeRegions,
                                                    dRegionsIntegral,
                                                    dRegionsError,
                                                    dParentsIntegral,
                                                    dParentsError,
                                                    subDividingDimension,
                                                    scannedArray,
                                                    newActiveRegions,
                                                    newActiveRegionsLength,
                                                    newActiveRegionsBisectDim,
                                                    numRegions,
                                                    numActiveRegions,
                                                    numOfDivisionOnDimension);

        if (VERBOSE) {
          Println(log, "\nCalling GPU Function divideIntervalsGPU");
          sprintf(msg,
                  "\n\t# of input intervals\t\t: %lu\n\t#. of division on "
                  "dimension\t: %i\n\t#. of Thread Blocks\t\t: %ld\n\t#. of "
                  "Threads per Blocks\t: %ld",
                  numActiveRegions,
                  numOfDivisionOnDimension,
                  numBlocks,
                  numThreads);
          Println(log, msg);
        }

        T *genRegions = 0, *genRegionsLength = 0;
        numBlocks = numActiveRegions / numThreads +
                    ((numActiveRegions % numThreads) ? 1 : 0);

        QuadDebug(cudaMalloc((void**)&genRegions,
                             sizeof(T) * numActiveRegions * NDIM *
                               numOfDivisionOnDimension));
        QuadDebug(cudaMalloc((void**)&genRegionsLength,
                             sizeof(T) * numActiveRegions * NDIM *
                               numOfDivisionOnDimension));

        divideIntervalsGPU<T>
          <<<numBlocks, numThreads>>>(genRegions,
                                      genRegionsLength,
                                      newActiveRegions,
                                      newActiveRegionsLength,
                                      newActiveRegionsBisectDim,
                                      numActiveRegions,
                                      numOfDivisionOnDimension);

        QuadDebug(Device.ReleaseMemory(newActiveRegions));
        QuadDebug(Device.ReleaseMemory(newActiveRegionsLength));
        QuadDebug(Device.ReleaseMemory(newActiveRegionsBisectDim));

        numRegions = numActiveRegions * numOfDivisionOnDimension;

        QuadDebug(Device.ReleaseMemory((void*)dRegions));
        QuadDebug(Device.ReleaseMemory((void*)dRegionsLength));
        QuadDebug(Device.ReleaseMemory((void*)scannedArray));

        dRegions = genRegions;
        dRegionsLength = genRegionsLength;
        cudaDeviceSynchronize();
       
        cudaDeviceSynchronize();
        // TODO: throws error
        // QuadDebug(cudaMemcpy(dRegions, 		genRegions, sizeof(T) *
        // numRegions * NDIM, cudaMemcpyDeviceToDevice));
        // QuadDebug(cudaMemcpy(dRegionsLength, 	genRegionsLength,
        // sizeof(T)
        // * numRegions * NDIM, cudaMemcpyDeviceToDevice));
      } else {
        numRegions = 0;
      }
    }

    void
    FirstPhaseIteration(T epsrel,
                         T epsabs,
                         T& integral,
                         T& error,
                         size_t& nregions,
                         size_t& neval,
                         T*& dParentsIntegral,
                         T*& dParentsError)
    {

      if (VERBOSE) {
        printf("===================================\n");
      }

      size_t numThreads = BLOCK_SIZE;
      size_t numBlocks = numRegions;

      T *dRegionsError = 0, *dRegionsIntegral = 0;
      T* newErrs = 0;

      if (VERBOSE) {
        printf(
          "Beginning of FirstPhaseIteration:: Allocating for %lu bad regions\n",
          numRegions * 2);
      }

      QuadDebug(Device.AllocateMemory((void**)&dRegionsIntegral,
                                      sizeof(T) * numRegions * 2));
      QuadDebug(Device.AllocateMemory((void**)&dRegionsError,
                                      sizeof(T) * numRegions * 2));

      if (numRegions == 1 && error == 0) {
        QuadDebug(Device.AllocateMemory((void**)&dParentsIntegral,
                                        sizeof(T) * numRegions * 2));
        QuadDebug(Device.AllocateMemory((void**)&dParentsError,
                                        sizeof(T) * numRegions * 2));
      }

      int *activeRegions = 0, *subDividingDimension = 0;

      if (VERBOSE) {
        printf("FirstPhaseIteration:: Currently have %lu bad regions\n",
               numRegions);
      }

      QuadDebug(Device.AllocateMemory((void**)&activeRegions,
                                      sizeof(int) * numRegions));
      QuadDebug(Device.AllocateMemory((void**)&subDividingDimension,
                                      sizeof(int) * numRegions));

      if (VERBOSE) {
        Println(log, "\nEntering function IntegrateFirstPhase \n");
        sprintf(msg,
                "\t# of input intervals\t\t: %ld\n\t#. of Thread Blocks\t\t: "
                "%ld\n\t#. of Threads per Blocks\t: %ld\n",
                numRegions,
                numBlocks,
                numThreads);
        Println(log, msg);
      }

      INTEGRATE_GPU_PHASE12<T><<<numBlocks, numThreads>>>(dRegions,
                                                          dRegionsLength,
                                                          numRegions,
                                                          dRegionsIntegral,
                                                          dRegionsError,
                                                          dParentsIntegral,
                                                          dParentsError,
                                                          activeRegions,
                                                          subDividingDimension,
                                                          epsrel,
                                                          epsabs,
                                                          constMem,
                                                          Rule.GET_FEVAL(),
                                                          Rule.GET_NSETS());

      QuadDebug(
        Device.AllocateMemory((void**)&newErrs, sizeof(T) * numRegions * 2));
      cudaDeviceSynchronize();

      if (numRegions != 1) {
        RefineError<T><<<numBlocks, numThreads>>>(dRegionsIntegral,
                                                  dRegionsError,
                                                  dParentsIntegral,
                                                  dParentsError,
                                                  newErrs,
                                                  activeRegions,
                                                  numRegions,
                                                  epsrel,
                                                  epsabs);
        cudaDeviceSynchronize();
        QuadDebug(cudaMemcpy(dRegionsError,
                             newErrs,
                             sizeof(T) * numRegions * 2,
                             cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();
      }

      nregions += numRegions;
      neval += numRegions * fEvalPerRegion;

      if (VERBOSE) {
        printf("computing the integral/error for %lu regions\n", numRegions);
      }

      // integral && error are the accumalated ones
      // we temporarily add the leaves to see what's happening

      thrust::device_ptr<T> wrapped_ptr;

      wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral + numRegions);
      T rG = integral + thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);

      wrapped_ptr = thrust::device_pointer_cast(dRegionsError + numRegions);
      T errG = error + thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);

      wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral);
      integral =
        integral + thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);

      wrapped_ptr = thrust::device_pointer_cast(dRegionsError);
      error = error + thrust::reduce(wrapped_ptr, wrapped_ptr + numRegions);

      if (VERBOSE) {
        printf("rG:%f\t errG:%f\t | global results: integral:%f\t error:%f\n",
               rG,
               errG,
               integral,
               error);
      }

      if ((errG <= MaxErr(rG, epsrel, epsabs)) && GLOBAL_ERROR) {

        if (VERBOSE) {
          sprintf(msg,
                  "Global Error Check -\t%ld integrand evaluations so far\n%lf "
                  "+- %lf ",
                  neval,
                  rG,
                  errG);
          Println(log, msg);
        }

        integral = rG;
        error = errG;
        numRegions = 0;
        return;
      }

      GenerateActiveIntervals(activeRegions,
                               subDividingDimension,
                               dRegionsIntegral,
                               dRegionsError,
                               dParentsIntegral,
                               dParentsError);

      if (VERBOSE) {
        printf("rG:%f\t errG:%f\t | global results: integral:%f\t error:%f\n",
               rG,
               errG,
               integral,
               error);
      }

      QuadDebug(cudaFree(subDividingDimension));

      QuadDebug(cudaFree(newErrs));
      QuadDebug(cudaFree(activeRegions));
      QuadDebug(cudaFree(dRegionsError));
      QuadDebug(cudaFree(dRegionsIntegral));
    }

    void
    IntegrateFirstPhase(T epsrel,
                        T epsabs,
                        T& integral,
                        T& error,
                        size_t& nregions,
                        size_t& neval)
    {

      T *dParentsError = 0, *dParentsIntegral = 0;

      for (int i = 0; i < 100; i++) {

        FirstPhaseIteration(epsrel,
                             epsabs,
                             integral,
                             error,
                             nregions,
                             neval,
                             dParentsIntegral,
                             dParentsError);
       
        if (numRegions < 1) {
          printf("NO BAD SUBREGIONS LEFT\n");
          return;
        }
        // printf("FIRST_PHASE_MAXREGIONS:%i\n", FIRST_PHASE_MAXREGIONS);
        if (numRegions >= FIRST_PHASE_MAXREGIONS) {
          printf("Reached the limit on Phase 1 regions supported (%i)\n",
                 FIRST_PHASE_MAXREGIONS);
          break;
        }
      }

      QuadDebug(cudaFree(dParentsIntegral));
      QuadDebug(cudaFree(dParentsError));

      hRegions =
        (T*)Host.AllocateMemory(&hRegions, sizeof(T) * numRegions * NDIM);
      hRegionsLength =
        (T*)Host.AllocateMemory(&hRegionsLength, sizeof(T) * numRegions * NDIM);
      QuadDebug(cudaMemcpy(hRegions,
                           dRegions,
                           sizeof(T) * numRegions * NDIM,
                           cudaMemcpyDeviceToHost));
      QuadDebug(cudaMemcpy(hRegionsLength,
                           dRegionsLength,
                           sizeof(T) * numRegions * NDIM,
                           cudaMemcpyDeviceToHost));
    }

    int
    IntegrateSecondPhase(T epsrel,
                         T epsabs,
                         T& integral,
                         T& error,
                         size_t& nregions,
                         size_t& neval,
                         T* optionalInfo = 0)
    {

      int numFailedRegions = 0;
      int num_gpus = 0; // number of CUDA GPUs

      if (optionalInfo != 0) {
        optionalInfo[0] = -INFTY;
      }

      /////////////////////////////////////////////////////////////////
      // determine the number of CUDA capable GPUs
      //
      cudaGetDeviceCount(&num_gpus);
      if (num_gpus < 1) {
        fprintf(stderr, "no CUDA capable devices were detected\n");
        exit(1);
      }
      int num_cpu_procs = omp_get_num_procs();

      /*
    Why did you have this section?
      for(int i = 1; i < num_gpus; i++){
    int gpu_id;
    QuadDebug(cudaSetDevice(i));	// "% num_gpus" allows more CPU threads
    than GPU devices QuadDebug(cudaGetDevice(&gpu_id));
    QuadDebug(cudaDeviceReset());
      }
      */

      if (VERBOSE) {
        /////////////////////////////////////////////////////////////////
        // display CPU and GPU configuration
        sprintf(msg, "number of host CPUs:\t%d\n", omp_get_num_procs());
        printf("number of host CPUs:\t%d\n", omp_get_num_procs());
        Println(log, msg);
        sprintf(msg, "number of CUDA devices:\t%d\n", num_gpus);
        printf("number of CUDA devices:\t%d\n", num_gpus);
        Println(log, msg);
        for (int i = 0; i < num_gpus; i++) {
          cudaDeviceProp dprop;
          cudaGetDeviceProperties(&dprop, i);
          sprintf(msg, "   %d: %s\n", i, dprop.name);
          Println(log, msg);
        }
        Println(log, "---------------------------\n");
      }

      // this works ok, check command line arg with actual devices available
      if (NUM_DEVICES > num_gpus)
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

        QuadDebug(cudaSetDevice(
          cpu_thread_id %
          num_gpus)); // "% num_gpus" allows more CPU threads than GPU devices
        QuadDebug(cudaGetDevice(&gpu_id));
        warmUpKernel<<<FIRST_PHASE_MAXREGIONS, BLOCK_SIZE>>>();

        if (VERBOSE) {
          sprintf(msg,
                  "CPU thread %d (of %d) uses CUDA device %d\n",
                  cpu_thread_id,
                  num_cpu_threads,
                  gpu_id);
          Println(log, msg);
        }

        if (cpu_thread_id < num_cpu_threads) {

          size_t numRegionsThread = numRegions / num_cpu_threads;
          int startIndex = cpu_thread_id * numRegionsThread;
          int endIndex = (cpu_thread_id + 1) * numRegionsThread;
          if (cpu_thread_id == (num_cpu_threads - 1))
            endIndex = numRegions;

          numRegionsThread = endIndex - startIndex;

          if (VERBOSE) {
            printf(
              "Num Regions in Phase 2:%lu (%u)\n", numRegions, cpu_thread_id);
            printf("Number of Regions going to each GPU:%lu (%u)\n",
                   numRegionsThread,
                   cpu_thread_id);
            printf("startIndex:%i (%i)\n", startIndex, cpu_thread_id);
            printf("endIndex:%i (%i)\n", endIndex, cpu_thread_id);
          }
          // QuadDebug(Device.SetHeapSize());
          CudaCheckError();

          Rule.loadDeviceConstantMemory(&constMem, cpu_thread_id);
          size_t numThreads = BLOCK_SIZE;
          size_t numBlocks = numRegionsThread;

          T *dRegionsError = 0, *dRegionsIntegral = 0;
          T *dRegionsThread = 0, *dRegionsLengthThread = 0;

          QuadDebug(Device.AllocateMemory((void**)&dRegionsIntegral,
                                          sizeof(T) * numRegionsThread));
          QuadDebug(Device.AllocateMemory((void**)&dRegionsError,
                                          sizeof(T) * numRegionsThread));

          int *activeRegions = 0, *subDividingDimension = 0,
              *dRegionsNumRegion = 0;

          QuadDebug(Device.AllocateMemory((void**)&activeRegions,
                                          sizeof(int) * numRegionsThread));
          QuadDebug(Device.AllocateMemory((void**)&subDividingDimension,
                                          sizeof(int) * numRegionsThread));
          QuadDebug(Device.AllocateMemory((void**)&dRegionsNumRegion,
                                          sizeof(int) * numRegionsThread));
          QuadDebug(Device.AllocateMemory((void**)&dRegionsThread,
                                          sizeof(T) * numRegionsThread * NDIM));
          QuadDebug(Device.AllocateMemory((void**)&dRegionsLengthThread,
                                          sizeof(T) * numRegionsThread * NDIM));

          CudaCheckError();
          // NOTE:Copy order is important

          for (int dim = 0; dim < NDIM; ++dim) {
            /*printf("copying from hRegions(%i-%i) -> dRegions(%i-%i) (%i)
               |numRegionsThread:%i numRegions:%i\n", dim * numRegions +
               startIndex , dim * numRegions + startIndex+numRegionsThread, dim
               * numRegionsThread, dim * numRegionsThread+numRegionsThread,
                                                                                                                                                                    cpu_thread_id, numRegionsThread, numRegions);*/
            QuadDebug(cudaMemcpy(dRegionsThread + dim * numRegionsThread,
                                 hRegions + dim * numRegions + startIndex,
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

          if (VERBOSE) {
            Println(log, "\n GPU Function PHASE2");
            sprintf(msg,
                    "\t# of input intervals\t\t: %ld\n\t#. of Thread "
                    "Blocks\t\t: %ld\n\t#. of Threads per Blocks\t: %ld\n",
                    numRegionsThread,
                    numBlocks,
                    numThreads);
            // printf(msg, "\t# of input intervals\t\t: %ld\n\t#. of Thread
            // Blocks\t\t: %ld\n\t#. of Threads per Blocks\t:
            // %ld\n",numRegionsThread, numBlocks, numThreads);
            Println(log, msg);
          }
          CudaCheckError();

          // std::cout << " phase2 : 	blocks:" << numBlocks << " threads:" <<
          // numThreads << std::endl; printf("Status before entering phase 2
          // %.12f +- %.12f\n", integral, error);
          cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2 * 128 * 1024 * 1024);

          double* exitCondition = nullptr;
          // QuadDebug(Device.AllocateMemory((void **)&exitCondition,
          // sizeof(double)*2)); cudaMemcpy(&exitCondition[0], &integral,
          // sizeof(T),	cudaMemcpyHostToDevice); cudaMemcpy(&exitCondition[1],
          // &error, 		sizeof(T),	cudaMemcpyHostToDevice);

          BLOCK_INTEGRATE_GPU_PHASE2<T>
            <<<numBlocks, numThreads, 0, stream[gpu_id]>>>(dRegionsThread,
                                                           dRegionsLengthThread,
                                                           numRegionsThread,
                                                           dRegionsIntegral,
                                                           dRegionsError,
                                                           dRegionsNumRegion,
                                                           activeRegions,
                                                           subDividingDimension,
                                                           epsrel,
                                                           epsabs,
                                                           gpu_id,
                                                           constMem,
                                                           Rule.GET_FEVAL(),
                                                           Rule.GET_NSETS(),
                                                           exitCondition);

          cudaDeviceSynchronize();
          // printf("BLOCK INTEGRATE_GPU done %d gpu:%i\n", cpu_thread_id,
          // gpu_id);
          CudaCheckError();
          // printf("After error checking and sync %d\n", cpu_thread_id);
          cudaDeviceSynchronize();
          cudaEventRecord(event[gpu_id], stream[gpu_id]);
          cudaEventSynchronize(event[gpu_id]);

          float elapsed_time;
          cudaEventElapsedTime(&elapsed_time, start, event[gpu_id]);

          if (optionalInfo != 0 && elapsed_time > optionalInfo[0]) {
            optionalInfo[0] = elapsed_time;
          }

          if (VERBOSE) {
            sprintf(msg,
                    "\nSecond Phase Kernel by thread %d (of %d) using CUDA "
                    "device %d took %.1f ms ",
                    cpu_thread_id,
                    num_cpu_threads,
                    gpu_id,
                    elapsed_time);
            Println(log, msg);
          }

          cudaEventDestroy(start);
          cudaEventDestroy(event[gpu_id]);

          thrust::device_ptr<T> wrapped_ptr;
          wrapped_ptr = thrust::device_pointer_cast(dRegionsIntegral);
          T integResult =
            thrust::reduce(wrapped_ptr, wrapped_ptr + numRegionsThread);
          // printf("integral %.12f + result %.12f\n", integral, integResult);
          integral += integResult;

          wrapped_ptr = thrust::device_pointer_cast(dRegionsError);
          error =
            error + thrust::reduce(wrapped_ptr, wrapped_ptr + numRegionsThread);

          thrust::device_ptr<int> int_ptr =
            thrust::device_pointer_cast(dRegionsNumRegion);
          int regionCnt = thrust::reduce(int_ptr, int_ptr + numRegionsThread);
          nregions += regionCnt;
          // std::cout << "Num regions : " << regionCnt << std::endl;

          neval += (regionCnt - numRegionsThread) * fEvalPerRegion * 2 +
                   numRegionsThread * fEvalPerRegion;

          int_ptr = thrust::device_pointer_cast(activeRegions);
          numFailedRegions +=
            thrust::reduce(int_ptr, int_ptr + numRegionsThread);

          //std::cout << "--" << numFailedRegions << std::endl;
          // QuadDebug(cudaThreadExit());

          QuadDebug(Device.ReleaseMemory(dRegionsError));
          QuadDebug(Device.ReleaseMemory(dRegionsIntegral));
          QuadDebug(Device.ReleaseMemory(dRegionsThread));
          QuadDebug(Device.ReleaseMemory(dRegionsLengthThread));
          QuadDebug(Device.ReleaseMemory(activeRegions));
          QuadDebug(Device.ReleaseMemory(subDividingDimension));
          QuadDebug(Device.ReleaseMemory(dRegionsNumRegion));
          QuadDebug(cudaDeviceSynchronize());
        } else
          printf("Rogue cpu thread\n");
      }

      // sprintf(msg, "Execution time : %.2lf", optionalInfo[0]);
      // Print(msg);
      return numFailedRegions;
    }
  };

}
#endif
