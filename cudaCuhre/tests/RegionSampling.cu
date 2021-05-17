#include "catch2/catch.hpp"
#include "demos/function.cuh"
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace quad;

class PTest {
public:   
    __device__ __host__ 
    double operator()(double x, double y){
        double res = 15.37;
        return res;
    }
};

class NTest {
public:   
    __device__ __host__ 
    double operator()(double x, double y){
        double res = -15.37;
        return res;
    }
};

class ZTest {
public:   
    __device__ __host__ 
    double operator()(double x, double y){
        return 0.;
    }
};

template<int NDIM>
void PrepKernel(quad::Kernel<double, NDIM>* kernel){
    int KEY = 0, VERBOSE = 0, heuristicID = 0, phase1type = 0, numDevices = 1, Final = 1;
    kernel->InitKernel(KEY, VERBOSE, numDevices);
    kernel->SetFinal(Final);
    kernel->SetVerbosity(VERBOSE);
    kernel->SetPhase_I_type(phase1type);
    kernel->SetHeuristicID(heuristicID);
    kernel->GenerateInitialRegions();
    
}

template<typename T>
void  CopyToHost(T*& cpuArray, T* gpuArray, size_t size){
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

namespace detail{
    struct Result{
        double estimate;
        double errorest;
    };
}

template <class K, size_t numArrays>
void
PrintGPUArrays(size_t arraySize, ...)
{
    va_list params;
    va_start(params, arraySize);
        
    std::array<K*, numArrays> hArrays;
        
    for(auto &array:hArrays){
        array = (K*)malloc(sizeof(K) * arraySize);
        cudaMemcpy(array, va_arg(params, K*), sizeof(K) * arraySize, cudaMemcpyDeviceToHost);
    }
        
    va_end(params);
        
    auto PrintSameIndex = [hArrays](size_t index){
        for(auto &array:hArrays){
            std::cout<< array[index] << "\t";
        }   
        std::cout << std::endl;
    };
        
    for(size_t i=0; i<arraySize; ++i)
        PrintSameIndex(i); 
        
    for(auto &array:hArrays)
        free(array);
}

template<typename IntegT, int NDIM, int BLOCKDIM>
detail::Result EvaluateRegions(quad::Kernel<double, NDIM>* kernel, IntegT* d_integrand){
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
    
    kernel->GetVars(numFuncEvals, numRegions, constMemPtr, nsets, depthBeingProcessed);
    kernel->GetPtrsToArrays(dRegions, dRegionsLength, dRegionsIntegral, dRegionsError, generators);
    kernel->IterationAllocations(dRegionsIntegral, dRegionsError, dParentsIntegral, dParentsError, activeRegions, subDividingDimension, iteration);
    
    QuadDebug(Device.AllocateMemory((void**)&generators,
                    sizeof(double) * NDIM * numFuncEvals));
                    
    ComputeGenerators<NDIM> <<<1, BLOCKDIM>>>(generators, numFuncEvals, *constMemPtr);   
    cudaDeviceSynchronize(); 
      
    Volume<double, NDIM> tempVol;
    cudaMalloc((void**)&lows, sizeof(double) * NDIM);
    cudaMalloc((void**)&highs, sizeof(double) * NDIM);
    cudaMemcpy(
          lows, tempVol.lows, sizeof(double) * NDIM, cudaMemcpyHostToDevice);
        cudaMemcpy(
          highs, tempVol.highs, sizeof(double) * NDIM, cudaMemcpyHostToDevice);  
    
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
          numFuncEvals,
          nsets,
          lows,
          highs,
          iteration,
          depthBeingProcessed,
          generators);
    cudaDeviceSynchronize(); 
    
    double* regionsIntegral = nullptr;
    double* regionsError = nullptr;
    detail::Result result;
    
    CopyToHost(regionsIntegral, dRegionsIntegral, numRegions);
    result.estimate = std::accumulate(regionsIntegral , regionsIntegral+numRegions, result.estimate);
    
    CopyToHost(regionsError, dRegionsError, numRegions);
    result.errorest = std::accumulate(regionsError , regionsError + numRegions, result.errorest);
    
    PrintGPUArrays<double, 1>(numRegions, dRegionsIntegral);
    printf("--\n");
    PrintGPUArrays<double, 1>(numRegions, dRegionsError);
    
    delete[] regionsError;
    delete[] regionsIntegral;
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
        result = EvaluateRegions<PTest, ndim, block_size>(&kernel, gpu_invocable_integrand);
        CHECK(abs(result.estimate - 15.37) <= .00000000000001);
        std::cout<<"error for 256 threads "<< result.errorest;
    }
    
    SECTION("128 Threads per Block")
    {
        constexpr int block_size = 128;
        result = EvaluateRegions<PTest, ndim, block_size>(&kernel, gpu_invocable_integrand);
        CHECK(abs(result.estimate - 15.37) <= .00000000000001);
        std::cout<<"error for 128 threads "<< result.errorest;
    }
    
    SECTION("64 Threads per Block")
    {
        constexpr int block_size = 64;
        result = EvaluateRegions<PTest, ndim, block_size>(&kernel, gpu_invocable_integrand);
        CHECK(abs(result.estimate - 15.37) <= .00000000000001);
        std::cout<<"error for 64 threads "<< result.errorest;
    }
}

TEST_CASE("Constant Negative Value Function")
{
    /*constexpr int ndim = 2;
    size_t numRegions = 16;
    NTest integrand;
    size_t maxIters = 1;
	int heuristicID = 0; 
    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    int key = 0;
    int verbose = 0;
    int numDevices = 1;
    Cuhre<double, 2> cuhre(0, nullptr, key, verbose, numDevices);
    cuhreResult res = cuhre.integrate<NTest>(integrand, epsrel, epsabs);
        
    double integral = res.estimate;
    double error = res.errorest;

    //returns are never precisely equal to 0. and -15.37
	printf("totalEstimate:%.15f\n", integral);
    CHECK(abs(integral - (-15.37)) <= .00000000000001);*/
}

TEST_CASE("Constant Zero Value Function")
{
    /*constexpr int ndim = 2;
    size_t numRegions = 16;
    ZTest integrand;
    size_t maxIters = 1;
	int heuristicID = 0; 
    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    int key = 0;
    int verbose = 0;
    int numDevices = 1;
    Cuhre<double, 2> cuhre(0, nullptr, key, verbose, numDevices);
    cuhreResult res = cuhre.integrate<ZTest>(integrand, epsrel, epsabs);
        
    double integral = res.estimate;
    double error = res.errorest;
    
    CHECK(integral== 0.0);*/
}

