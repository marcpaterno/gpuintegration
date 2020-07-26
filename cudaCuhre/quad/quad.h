#ifndef CUDACUHRE_QUAD_QUAD_h
#define CUDACUHRE_QUAD_QUAD_h

#define TIMING_DEBUG 1
#define BLOCK_SIZE 256
#define SM_REGION_POOL_SIZE 128

#define GLOBAL_ERROR 1
#define MAX_GLOBALPOOL_SIZE 2048

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
  
  ~Structures(){
	
  }

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

struct cuhreResult{
	double value;
	double error;
	size_t neval;
	size_t nregions;
	bool   status;
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

template<int dim>
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

#endif
