#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/demos/function.cuh"
#include "cuda/pagani/quad/GPUquad/Pagani.cuh"
#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/Volume.cuh"
#include "cuda/pagani/quad/util/cudaUtil.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace quad;

class PTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    double res = 15.37;
    return res;
  }
};

class NTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    double res = -15.37;
    return res;
  }
};

class ZTest {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    return 0.;
  }
};

template <int NDIM>
void
PrepKernel(quad::Kernel<double, NDIM>* kernel)
{
  int KEY = 0, VERBOSE = 0, heuristicID = 0, numDevices = 1, Final = 1;
  kernel->InitKernel(KEY, VERBOSE, numDevices);
  kernel->SetFinal(Final);
  kernel->SetVerbosity(VERBOSE);
  kernel->SetHeuristicID(heuristicID);
  kernel->GenerateInitialRegions();
}

template <typename T>
void
CopyToHost(T*& cpuArray, T* gpuArray, size_t size)
{
  cpuArray = new T[size];
  cudaMemcpy(cpuArray, gpuArray, sizeof(T) * size, cudaMemcpyDeviceToHost);
}

template <typename IntegT>
IntegT*
Make_GPU_Integrand(IntegT* integrand)
{
  IntegT* d_integrand;
  cudaMallocManaged((void**)&d_integrand, sizeof(IntegT));
  memcpy(d_integrand, &integrand, sizeof(IntegT));
  return d_integrand;
}

namespace detail {
  struct Result {
    double estimate = 0.;
    double errorest = 0.;
  };
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

template <typename IntegT, int NDIM, int BLOCKDIM>
detail::Result
EvaluateRegions(quad::Kernel<double, NDIM>* kernel, IntegT* d_integrand)
{
  int iteration = 0, nsets = 0, depthBeingProcessed = 0;
  double* dRegionsIntegral = nullptr;
  double* dRegionsError = nullptr;
  double* dParentsIntegral = nullptr;
  double* dParentsError = nullptr;
  double* dRegions = nullptr;
  double* dRegionsLength = nullptr;
  double* generators = nullptr;
  double* highs = nullptr;
  double* lows = nullptr;

  int *activeRegions = 0, *subDividingDimension = 0;
  size_t numRegions = 0, numFuncEvals = 0;

  double epsrel = 1.e-3;
  double epsabs = 1.e-12;

  DeviceMemory<double> Device;
  Structures<double>* constMemPtr = nullptr;

  kernel->GetVars(
    numFuncEvals, numRegions, constMemPtr, nsets, depthBeingProcessed);
  CudaCheckError();
  kernel->GetPtrsToArrays(
    dRegions, dRegionsLength, dRegionsIntegral, dRegionsError, generators);
  CudaCheckError();
  kernel->IterationAllocations(dRegionsIntegral,
                               dRegionsError,
                               dParentsIntegral,
                               dParentsError,
                               activeRegions,
                               subDividingDimension,
                               iteration);
  CudaCheckError();

  QuadDebug(Device.AllocateMemory((void**)&generators,
                                  sizeof(double) * NDIM * numFuncEvals));
  CudaCheckError();
  ComputeGenerators<double, NDIM>
    <<<1, BLOCK_SIZE>>>(generators, numFuncEvals, *constMemPtr);
  CudaCheckError();
  CudaCheckError();

  Volume<double, NDIM> tempVol;

  cudaMalloc((void**)&lows, sizeof(double) * NDIM);
  cudaMalloc((void**)&highs, sizeof(double) * NDIM);
  cudaMemcpy(lows, tempVol.lows, sizeof(double) * NDIM, cudaMemcpyHostToDevice);
  cudaMemcpy(
    highs, tempVol.highs, sizeof(double) * NDIM, cudaMemcpyHostToDevice);
  CudaCheckError();

  kernel->AllocVolArrays(&tempVol);
  quad::Func_Evals<NDIM> fevals;
  INTEGRATE_GPU_PHASE1<IntegT, double, NDIM, BLOCKDIM>
    <<<numRegions, BLOCKDIM, NDIM * sizeof(GlobalBounds)>>>(
      d_integrand,
      dRegions,
      dRegionsLength,
      numRegions,
      dRegionsIntegral,
      dRegionsError,
      activeRegions,
      subDividingDimension,
      epsrel,
      epsabs,
      *constMemPtr,
      lows,
      highs,
      generators,
      fevals);
  cudaDeviceSynchronize();
  CudaCheckError();
  double* regionsIntegral = nullptr;
  double* regionsError = nullptr;
  detail::Result result;

  CopyToHost(regionsIntegral, dRegionsIntegral, numRegions);
  result.estimate = std::accumulate(
    regionsIntegral, regionsIntegral + numRegions, result.estimate);
  CudaCheckError();
  CopyToHost(regionsError, dRegionsError, numRegions);
  result.errorest =
    std::accumulate(regionsError, regionsError + numRegions, result.errorest);
  CudaCheckError();

  Device.ReleaseMemory(generators);
  delete[] regionsError;
  delete[] regionsIntegral;
  CudaCheckError();
  return result;
}

TEST_CASE("Constant Positive Value Function")
{
  constexpr int ndim = 2;
  PTest integrand;
  Kernel<double, ndim> kernel(std::cout);
  PrepKernel<ndim>(&kernel);
  PTest* gpu_invocable_integrand = Make_GPU_Integrand(&integrand);
  detail::Result result;

  SECTION("256 Threads per Block")
  {
    constexpr int block_size = 256;
    result = EvaluateRegions<PTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    std::cout << "result:" << result.estimate << std::endl;
    CHECK(abs(result.estimate - 15.37) <= .00000000000001);
  }

  SECTION("128 Threads per Block")
  {
    constexpr int block_size = 128;
    result = EvaluateRegions<PTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    CHECK(abs(result.estimate - 15.37) <= .00000000000001);
  }

  SECTION("64 Threads per Block")
  {
    constexpr int block_size = 64;
    result = EvaluateRegions<PTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    CHECK(abs(result.estimate - 15.37) <= .00000000000001);
  }
}

TEST_CASE("Constant Negative Value Function")
{
  constexpr int ndim = 2;
  NTest integrand;
  Kernel<double, ndim> kernel(std::cout);
  PrepKernel<ndim>(&kernel);
  NTest* gpu_invocable_integrand = Make_GPU_Integrand(&integrand);
  detail::Result result;

  SECTION("256 Threads per Block")
  {
    constexpr int block_size = 256;
    result = EvaluateRegions<NTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    CHECK(abs(result.estimate - (-15.37)) <= .00000000000001);
  }

  SECTION("128 Threads per Block")
  {
    constexpr int block_size = 128;
    result = EvaluateRegions<NTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    CHECK(abs(result.estimate - (-15.37)) <= .00000000000001);
  }

  SECTION("64 Threads per Block")
  {
    constexpr int block_size = 64;
    result = EvaluateRegions<NTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    CHECK(abs(result.estimate - (-15.37)) <= .00000000000001);
  }
}

TEST_CASE("Constant Zero Value Function")
{
  constexpr int ndim = 2;
  ZTest integrand;
  Kernel<double, ndim> kernel(std::cout);
  PrepKernel<ndim>(&kernel);
  ZTest* gpu_invocable_integrand = Make_GPU_Integrand(&integrand);
  detail::Result result;

  SECTION("256 Threads per Block")
  {
    constexpr int block_size = 256;
    result = EvaluateRegions<ZTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    CudaCheckError();
    CHECK(result.estimate == 0.0);
  }

  SECTION("128 Threads per Block")
  {
    constexpr int block_size = 128;
    result = EvaluateRegions<ZTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    CudaCheckError();
    CHECK(result.estimate == 0.0);
  }

  SECTION("64 Threads per Block")
  {
    constexpr int block_size = 64;
    result = EvaluateRegions<ZTest, ndim, block_size>(&kernel,
                                                      gpu_invocable_integrand);
    CudaCheckError();
    CHECK(result.estimate == 0.0);
  }
}
