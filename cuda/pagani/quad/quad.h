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

#include <fstream>
#include <string>
#include <vector>
#include "cuda/pagani/quad/util/cuhreResult.cuh"
using TYPE = double;

static int FIRST_PHASE_MAXREGIONS = (1 << 14);
__constant__ TYPE errcoeff[] = {5, 1, 5};

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

  T* _gpuG;
  T* _cRuleWt;
  T* _GPUScale;
  T* _GPUNorm;
  int* _gpuGenPos;
  int* _gpuGenPermGIndex;
  int* _gpuGenPermVarCount;
  int* _gpuGenPermVarStart;
  size_t* _cGeneratorCount;
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
  double avg, err;
  int bisectdim;
};

struct Bounds {
  double lower, upper;
};

struct GlobalBounds {
  double unScaledLower, unScaledUpper;
};

struct PhaseII_output {
  // perhaps activeRegions belongs here based on what attributes Region has
  PhaseII_output()
  {
    estimate = 0.;
    errorest = 0.;
    regions = 0;
    num_failed_blocks = 0;
    num_starting_blocks = 0;
  }

  double estimate;
  double errorest;
  int regions;
  int num_failed_blocks;
  int num_starting_blocks;

  PhaseII_output
  operator+(const PhaseII_output& b)
  {
    PhaseII_output addedObj;
    addedObj.estimate = estimate + b.estimate;
    addedObj.errorest = errorest + b.errorest;
    addedObj.regions = regions + b.regions;
    addedObj.num_failed_blocks = num_failed_blocks + b.num_failed_blocks;
    addedObj.num_starting_blocks = regions + b.num_starting_blocks;
    return addedObj;
  }

  void
  operator=(const PhaseII_output& b)
  {
    estimate = b.estimate;
    errorest = b.errorest;
    regions = b.regions;
    num_failed_blocks = b.num_failed_blocks;
    num_starting_blocks = b.num_starting_blocks;
  }

  void
  operator+=(const PhaseII_output& b)
  {
    estimate += b.estimate;
    errorest += b.errorest;
    regions += b.regions;
    num_failed_blocks += b.num_failed_blocks;
    num_starting_blocks += b.num_starting_blocks;
  }
};

template <int dim>
struct Region {
  int div;
  Result result;
  Bounds bounds[dim];
};

#define NRULES 5

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
