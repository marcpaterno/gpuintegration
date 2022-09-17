#ifndef CUDACUHRE_QUAD_QUAD_h
#define CUDACUHRE_QUAD_QUAD_h

#define TIMING_DEBUG 1
//#define BLOCK_SIZE 256
//#define BLOCK_SIZE 128
#define BLOCK_SIZE 64
//#define BLOCK_SIZE 32
#define SM_REGION_POOL_SIZE 128

#define GLOBAL_ERROR 1
#define MAX_GLOBALPOOL_SIZE 2048

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <fstream>
#include <string>
#include <vector>
#include "oneAPI/pagani/quad/util/cuhreResult.dp.hpp"
using TYPE = double;

static int FIRST_PHASE_MAXREGIONS = (1 << 14);
//dpct::constant_memory<TYPE, 1> errcoeff(sycl::range(1), {5, 1, 5});

// Utilities
#include "util/cudaArchUtil.h"
#include "util/cudaDebugUtil.h"

class VerboseResults{
	public:
		std::vector<std::vector<double>> funcEvaluationPoints;
		std::vector<double> results;
		size_t numFuncEvals = 0;
		size_t NDIM = 0;
};

template <typename T>
struct Structures {
  
  Structures() = default;
   /* : _gpuG(nullptr)
    , _cRuleWt(nullptr)
    , _GPUScale(nullptr)
    , _gpuGenPos(nullptr)
    , _gpuGenPermGIndex(nullptr)
    , _gpuGenPermVarCount(nullptr)
    , _gpuGenPermVarStart(nullptr)
    , _cGeneratorCount(nullptr)
    , _GPUNorm(nullptr)
  {}*/

  T* _gpuG = nullptr;
  T* _cRuleWt = nullptr;
  T* _GPUScale = nullptr;
  T* _GPUNorm = nullptr;
  int* _gpuGenPos = nullptr;
  int* _gpuGenPermGIndex = nullptr;
  int* _gpuGenPermVarCount = nullptr;
  int* _gpuGenPermVarStart = nullptr;
  size_t* _cGeneratorCount = nullptr;
};

/*template <typename T>
struct cuhreResult {
  T estimate {};
  T errorest {};
  size_t neval = 0;
  size_t nregions = 0;
  size_t nFinishedRegions = 0;
  int status = -1;
  int lastPhase = 0;
  // size_t activeRegions;    // is not currently being set
  size_t phase2_failedblocks = 0; // is not currently being set
  double chi_sq = 0.;
};*/

struct Result {
  double avg = 0., err = 0.;
  int bisectdim = 0;
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

#endif
