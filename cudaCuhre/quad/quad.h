#ifndef CUDACUHRE_QUAD_QUAD_h
#define CUDACUHRE_QUAD_QUAD_h

#define TIMING_DEBUG 1
#define BLOCK_SIZE 256
#define SM_REGION_POOL_SIZE 128

#define GLOBAL_ERROR 1
#define MAX_GLOBALPOOL_SIZE 2048

#include <fstream>
#include <sstream>
#include <string>

using TYPE = double;

static int FIRST_PHASE_MAXREGIONS = (1 << 14);
//static int FIRST_PHASE_MAXREGIONS = (1 << 15);
__constant__ TYPE errcoeff[] = {5, 1, 5};

// Utilities
#include "util/cudaArchUtil.h"
#include "util/cudaDebugUtil.h"
#include "GPUquad/Interp2D.cuh"

template <typename T>
struct Structures {
  __host__ __device__
  Structures()
    : _gpuG(nullptr)
    , _cRuleWt(nullptr)
    , _GPUScale(nullptr)
    , _gpuGenPos(nullptr)
    , _gpuGenPermGIndex(nullptr)
    , _gpuGenPermVarCount(nullptr)
    , _gpuGenPermVarStart(nullptr)
    , _cGeneratorCount(nullptr)
    , _GPUNorm(nullptr)
  {}

  ~Structures() {}

  T* /*const __restrict__*/ _gpuG;
  T* /*const __restrict__*/ _cRuleWt;
  T* /*const __restrict__*/ _GPUScale;
  T* /*const __restrict__*/ _GPUNorm;
  int* /*const __restrict__*/ _gpuGenPos;
  int* /*const __restrict__*/ _gpuGenPermGIndex;
  int* /*const __restrict__*/ _gpuGenPermVarCount;
  int* /*const __restrict__*/ _gpuGenPermVarStart;
  size_t* /*const __restrict__*/ _cGeneratorCount;
};

struct cuhreResult {

  cuhreResult()
  {
    estimate = 0.;
    errorest = 0.;
    neval = 0.;
    nregions = 0.;
    status = 0.;
    //activeRegions = 0.;
    phase2_failedblocks = 0.;
	lastPhase = 0;
  };

  double estimate;
  double errorest;
  size_t neval;
  size_t nregions;
  int status;
  int lastPhase;
  //size_t activeRegions;    // is not currently being set
  size_t phase2_failedblocks; // is not currently being set
};

struct Result {
  double avg, err;
  int bisectdim;
};

struct Bounds {
  double lower, upper;
};

struct GlobalBounds {
  double unScaledLower, unScaledUpper;
};

class Managed 
{
	 public:
	  void *operator new(size_t len) {
		void *ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	  }

	  void operator delete(void *ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	  }
};	

class RegionList: public Managed{
	//Deriving from “Managed” allows pass-by-reference
	public:
		RegionList()
		{
			ndim = 0;
			numRegions = 0;
			activeRegions = nullptr;
			subDividingDimension = nullptr;
			dRegionsLength = nullptr;
			dRegions = nullptr;
			dRegionsIntegral = nullptr;
			dRegionsError = nullptr;
		}
		
		RegionList(const int dim, const size_t size)
		{
			ndim = dim;
			numRegions = size;
			
			cudaMallocManaged(&activeRegions, 		 sizeof(int)*size);
			cudaMallocManaged(&subDividingDimension, sizeof(int)*size);
			cudaMallocManaged(&dRegionsIntegral, 	 sizeof(double)*size);
			cudaMallocManaged(&dRegionsError, 		 sizeof(double)*size);
			
			cudaMallocManaged(&dRegionsLength, 	sizeof(double)*size*ndim);
			cudaMallocManaged(&dRegions, 		sizeof(double)*size*ndim);
		}
		
		void UnifiedInit(const int dim, const size_t size)
		{
			//currently not required by kernel do be done this way
			ndim = dim;
			numRegions = size;
			
			cudaMallocManaged(&activeRegions, 		 sizeof(int)*size);
			cudaMallocManaged(&subDividingDimension, sizeof(int)*size);
			cudaMallocManaged(&dRegionsIntegral, 	 sizeof(double)*size);
			cudaMallocManaged(&dRegionsError, 		 sizeof(double)*size);
			
			cudaMallocManaged(&dRegionsLength, 	sizeof(double)*size*ndim);
			cudaMallocManaged(&dRegions, 		sizeof(double)*size*ndim);
		}
		
		void Init(const int dim, const size_t size)
		{
			//currently not required by kernel do be done this way
			ndim = dim;
			numRegions = size;
			
			cudaMalloc((void**)&activeRegions, 		  sizeof(int)*size);
			cudaMalloc((void**)&subDividingDimension, sizeof(int)*size);
			cudaMalloc((void**)&dRegionsIntegral, 	  sizeof(double)*size);
			cudaMalloc((void**)&dRegionsError, 		  sizeof(double)*size);
			
			cudaMalloc((void**)&dRegionsLength, 	 sizeof(double)*size*ndim);
			cudaMalloc((void**)&dRegions, 			 sizeof(double)*size*ndim);
		}
		
		void Clear(){
			cudaFree(activeRegions);
			cudaFree(subDividingDimension);
			cudaFree(dRegionsIntegral);
			cudaFree(dRegionsError);
			cudaFree(dRegionsLength);
			cudaFree(dRegions);
		}
		
		void Set(int dim, size_t num, double* regions, double* regionsLength, double* regions_integral, double* regions_err , int* nextDim, int *active)
		{
			ndim = dim;
			numRegions = num;
			activeRegions = active;
			subDividingDimension = nextDim;
			dRegionsLength = regionsLength;
			dRegions = regions;
			dRegionsIntegral = regions_integral;
			dRegionsError = regions_err;
		}			
		
		void Set(int dim, size_t num, double* regions, double* regionsLength, double* regions_integral, double* regions_err )
		{
			//this is used if you want to use a region list with externally alloacted memory
			ndim = dim;
			numRegions = num;
			dRegionsLength = regionsLength;
			dRegions = regions;
			dRegionsIntegral = regions_integral;
			dRegionsError = regions_err;
		}			
		
		RegionList (const RegionList &s) 
		{
			//Unified memory copy constructor allows pass-by-value
			ndim = s.ndim;
			numRegions = s.numRegions;
			cudaMallocManaged(&activeRegions, sizeof(int)*numRegions);
			cudaMallocManaged(&subDividingDimension, sizeof(int)*numRegions);
			cudaMallocManaged(&dRegionsLength, sizeof(double)*numRegions);
			cudaMallocManaged(&dRegions, sizeof(double)*numRegions);
			cudaMallocManaged(&dRegionsIntegral, sizeof(double)*numRegions);
			cudaMallocManaged(&dRegionsError, sizeof(double)*numRegions);
			
			memcpy(activeRegions, 		 s.activeRegions, sizeof(int)*numRegions);
			memcpy(subDividingDimension, s.subDividingDimension, sizeof(int)*numRegions);
			memcpy(dRegionsLength, 		 s.dRegionsLength, sizeof(double)*numRegions);
			memcpy(dRegions, 			 s.dRegions, sizeof(double)*numRegions);
			memcpy(dRegionsIntegral, 	 s.dRegionsIntegral, sizeof(double)*numRegions);
			memcpy(dRegionsError, 		 s.dRegionsError, sizeof(double)*numRegions);
		}
		
		double* dRegionsError;
		double* dRegionsIntegral;
		double* dRegions;
		double* dRegionsLength;
		int* 	subDividingDimension;
		int* 	activeRegions;
		
		int ndim;
		size_t numRegions;
};

template <int dim>
struct Region {
  int div;
  Result result;
  Bounds bounds[dim];
};

#define NRULES 5

extern __shared__ GlobalBounds sBound[];

__shared__ TYPE sdata[BLOCK_SIZE];
__shared__ TYPE* serror;
__shared__ size_t* serrorPos;
__shared__ bool GlobalMemCopy;
__shared__ int max_global_pool_size;
__shared__ TYPE ERR, RESULT;
__shared__ size_t gRegionPos[SM_REGION_POOL_SIZE / 2], gRegionPoolSize;

template <int NDIM>
struct Snapshot {
  __host__
  Snapshot(int* iterations, int size)
  {
    numSnapshots = size;
    total_regions = 0;
    currArrayHead = 0;

    for (int i = 0; i < size; i++)
      total_regions += iterations[i];

    cudaMalloc((void**)&arr, sizeof(Region<NDIM>) * total_regions);
    cudaMalloc((void**)&sizes, sizeof(int) * numSnapshots);
    cudaMemcpy(
      sizes, iterations, sizeof(int) * numSnapshots, cudaMemcpyHostToDevice);
  }

  __host__ void
  Save(std::string baseFileName)
  {

    Region<NDIM>* h_arr = 0;
    int* h_sizes = 0;

    h_arr = (Region<NDIM>*)malloc(sizeof(Region<NDIM>) * total_regions);
    h_sizes = (int*)malloc(sizeof(int) * numSnapshots);

    cudaMemcpy(
      h_arr, arr, sizeof(Region<NDIM>) * total_regions, cudaMemcpyDeviceToHost);
    cudaMemcpy(
      h_sizes, sizes, sizeof(int) * numSnapshots, cudaMemcpyDeviceToHost);
    int index = 0;

    for (int i = 0; i < numSnapshots; i++) {
      std::string filename = baseFileName + std::to_string(h_sizes[i]) + ".csv";
      std::ofstream outfile(filename.c_str());
      int current_size = h_sizes[i];
      int snapShotStartIndex = 0;

      for (int j = 0; j < i; j++)
        snapShotStartIndex += h_sizes[j];

      for (; index < current_size + snapShotStartIndex; index++) {
        outfile << h_arr[index].result.avg << "," << h_arr[index].result.err
                << ",";
        for (int dim = 0; dim < NDIM; dim++) {
          outfile << h_arr[index].bounds[dim].upper << ","
                  << h_arr[index].bounds[dim].lower << ",";
        }
        outfile << h_arr[index].div << "," << -1 << std::endl;
      }
      outfile.close();
    }

    free(h_arr);
    free(h_sizes);
    cudaFree(sizes);
    cudaFree(arr);
  }

  __host__ __device__
  Snapshot()
  {
    numSnapshots = 0;
    arr = nullptr;
    sizes = nullptr;
  }

  int currArrayHead;
  int numSnapshots;
  Region<NDIM>* arr;
  int* sizes;
  int total_regions;
};

#endif
