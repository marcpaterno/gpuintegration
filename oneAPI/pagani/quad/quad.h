#ifndef CUDACUHRE_QUAD_QUAD_h
#define CUDACUHRE_QUAD_QUAD_h

#define TIMING_DEBUG 1
#define BLOCK_SIZE 64
#define SM_REGION_POOL_SIZE 128

#define GLOBAL_ERROR 1
#define MAX_GLOBALPOOL_SIZE 2048

#include <CL/sycl.hpp>
#include <fstream>
#include <string>
#include <vector>
#include "common/oneAPI/cuhreResult.dp.hpp"
using TYPE = double;

static int FIRST_PHASE_MAXREGIONS = (1 << 14);

// Utilities
#include "common/oneAPI/cudaArchUtil.h"
#include "common/oneAPI/cudaDebugUtil.h"

class VerboseResults {
public:
  std::vector<std::vector<double>> funcEvaluationPoints;
  std::vector<double> results;
  size_t numFuncEvals = 0;
  size_t NDIM = 0;
};

template <typename T>
struct Structures {

  Structures() = default;

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
