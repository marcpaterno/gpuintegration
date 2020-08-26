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

__constant__ TYPE errcoeff[] = {5, 1, 5};

// Utilities
#include "util/cudaArchUtil.h"
#include "util/cudaDebugUtil.h"

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
	
  cuhreResult(){
	  estimate = 0.;
	  errorest = 0.;
	  neval = 0.;
	  nregions = 0.;
	  status = 0.;
	  activeRegions = 0.;
	  phase2_failedblocks = 0.;
  };
  
  double estimate;
  double errorest;
  size_t neval;
  size_t nregions;
  int status;
  size_t activeRegions;		//is not currently being set
  int phase2_failedblocks;  //is not currently being set
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
struct Snapshot{
	__host__ 
	Snapshot(int* iterations, int size){
		numSnapshots  = size;
		total_regions = 0;
		currArrayHead = 0;

		for(int i=0; i< size; i++)
			total_regions+=iterations[i];
		
		cudaMalloc((void**)&arr, sizeof(Region<NDIM>) * total_regions);
		cudaMalloc((void**)&sizes, sizeof(int) * numSnapshots);
		cudaMemcpy(sizes, iterations, sizeof(int) * numSnapshots, cudaMemcpyHostToDevice);
	}
		
	__host__ 
	void Save(std::string baseFileName){
		
		Region<NDIM>* h_arr = 0;
		int* h_sizes = 0;
		
		h_arr = (Region<NDIM>*)malloc(sizeof(Region<NDIM>) * total_regions);
		h_sizes = (int*)malloc(sizeof(int) * numSnapshots);
		
		cudaMemcpy(h_arr, 	arr, 	sizeof(Region<NDIM>) * total_regions, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_sizes, sizes, 	sizeof(int) * numSnapshots, cudaMemcpyDeviceToHost);
		int index = 0;
		
		for(int i=0; i< numSnapshots; i++){
			std::string filename = baseFileName + std::to_string(h_sizes[i]) + ".csv";
			std::ofstream outfile(filename.c_str());
			int current_size = h_sizes[i];
			int snapShotStartIndex = 0;
			
			for(int j=0; j<i; j++)
				snapShotStartIndex += h_sizes[j];
			
			for(; index < current_size + snapShotStartIndex; index++){
				outfile << h_arr[index].result.avg << "," << h_arr[index].result.err << ",";
					for(int dim = 0; dim < NDIM; dim++){
						outfile<<h_arr[index].bounds[dim].upper<<","<< h_arr[index].bounds[dim].lower<<",";
					}
					outfile <<h_arr[index].div<<","<<-1<<std::endl;
			}
			outfile.close();
		}

		free(h_arr);
		free(h_sizes);
		cudaFree(sizes);
		cudaFree(arr);
	}
	
	__host__ __device__ 
	Snapshot(){
		numSnapshots = 0; 
		arr   = nullptr; 
		sizes = nullptr;
	}
		
	int currArrayHead;
	int numSnapshots;
	Region<NDIM>* arr;
	int* sizes;
	int total_regions;
};

#endif
