#define CATCH_CONFIG_MAIN



#include "catch2/catch.hpp"
#include "cuda/cudaPagani/demos/function.cuh"
#include "cuda/cudaPagani/quad/GPUquad/Pagani.cuh"
#include "cuda/cudaPagani/quad/quad.h"
#include "cuda/cudaPagani/quad/util/Volume.cuh"
#include "cuda/cudaPagani/quad/util/cudaUtil.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace quad;

template <typename T>
void
CopyToHost(T*& cpuArray, T* gpuArray, size_t size)
{
  cpuArray = new T[size];
  cudaMemcpy(cpuArray, gpuArray, sizeof(T) * size, cudaMemcpyDeviceToHost);
}

template <class K, size_t numArrays>
void
PrintGPUArrays(size_t arraySize, ...)
{
  va_list params;
  va_start(params, arraySize);

  std::array<K*, numArrays> hArrays;

  for (auto& array : hArrays) {
    array = (K*)malloc(sizeof(K) * arraySize);
    cudaMemcpy(
      array, va_arg(params, K*), sizeof(K) * arraySize, cudaMemcpyDeviceToHost);
  }

  va_end(params);

  auto PrintSameIndex = [hArrays](size_t index) {
    for (auto& array : hArrays) {
      std::cout << array[index] << "\t";
    }
    std::cout << std::endl;
  };

  for (size_t i = 0; i < arraySize; ++i)
    PrintSameIndex(i);

  for (auto& array : hArrays)
    free(array);
}

__host__ __device__ int
FourthDividedDifference(double* sdata, int NDIM, int* maxdim)
{

  double ratio = 0.180444;
  int offset = 2 * NDIM;

  double* f = &sdata[0];
  double* f1 = f;
  double base = *f1 * 2 * (1 - ratio);
  double maxdiff = 0;
  int bisectdim = *maxdim;
  for (int dim = 0; dim < NDIM; ++dim) {
    double* fp = f1 + 1;
    double* fm = fp + 1;
    double fourthdiff =
      fabs(base + ratio * (fp[0] + fm[0]) - (fp[offset] + fm[offset]));
    f1 = fm;
    if (fourthdiff > maxdiff) {
      maxdiff = fourthdiff;
      bisectdim = dim;
    }
  }
  return bisectdim;
}

__host__ __device__ int
newFourthDividedDifference(double* sdata, int NDIM, int* maxdim)
{

  double ratio = 0.180444;
  int offset = 2 * NDIM;
  double base = sdata[0] * 2 * (1 - ratio);

  double maxdiff = 0;
  int index = 0;
  int bisectdim = *maxdim;
  for (int dim = 0; dim < NDIM; ++dim) {
    double fourthdiff =
      fabs(base + ratio * (sdata[index + 1] + sdata[index + 2]) -
           (sdata[index + 1 + offset] + sdata[index + 2 + offset]));
    if (fourthdiff > maxdiff) {
      maxdiff = fourthdiff;
      bisectdim = dim;
    }
    printf("For Dim:%i Accessing index:%i\n", dim, index + 2 + offset);
    index += 2;
  }
  return bisectdim;
}

int
GetMaxDim(double* regionLows, double* regionHighs, int NDIM)
{
  double maxRange = 0;
  int maxDim = 0;
  for (int dim = 0; dim < NDIM; ++dim) {
    double lower = regionLows[dim];
    double upper = lower + regionHighs[dim];
    double range = upper - lower;
    if (range > maxRange) {
      maxDim = dim;
      maxRange = range;
    }
  }
  return maxDim;
}

int funcevalsPerDIM[7] = {33, 77, 153, 273, 453, 717, 1105};

void
Fill_sdata(double* sdata, int NDIM)
{
  int funcevals = funcevalsPerDIM[NDIM - 1];
  for (int i = 0; i < funcevals; i++)
    sdata[i] = .1 + (double)i;
}

TEST_CASE("Check Across Dimensions 2-8")
{
  double highs[8] = {1., 2., 3., 4., 5., 6., 7., 8.};
  double lows[8] = {.1, .2, .3, .4, .5, .6, .7, .8};

  for (int dim = 8; dim >= 2; dim--) {
    int ndim = dim;
    int funcevals = funcevalsPerDIM[ndim - 1];
    double sdata[funcevals];
    int maxDim = GetMaxDim(lows + dim - 2,
                           highs + dim - 2,
                           ndim); // to avoid multiple highs, lows arrays
    Fill_sdata(sdata, ndim);
    int origResult = FourthDividedDifference(sdata, ndim, &maxDim);
    int newResult = newFourthDividedDifference(sdata, ndim, &maxDim);
    std::cout << origResult << " vs " << newResult << std::endl;
    CHECK(origResult == newResult);
  }
}
