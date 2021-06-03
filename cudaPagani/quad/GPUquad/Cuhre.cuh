#ifndef CUDACUHRE_QUAD_GPUQUAD_CUHRE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_CUHRE_CUH

#include "../util/cudaMemoryUtil.h"
#include "../util/cudaTimerUtil.h"

#include "../util/Volume.cuh"
#include "Kernel.cuh"
#include <chrono>
#include <map>
#include <stdio.h>

namespace quad {
#if TIMING_DEBUG == 1
  timer::event_pair timer_one;
#endif

  template <typename T, int NDIM>
  class Cuhre {

    // Debug message
    char msg[256];
    int KEY;

    int VERBOSE;
    int numDevices;

    int argc;
    char** argv;

    T epsrel, epsabs;
    Kernel<T, NDIM>* kernel;
    std::ofstream log;

  public:
    Cuhre(int pargc,
          char** pargv,
          int key = 0,
          int verbose = 0,
          int numDevices = 1)
    {
      // QuadDebug(cudaDeviceReset());
      argc = pargc;
      argv = pargv;
      KEY = key;
      VERBOSE = verbose;
      this->numDevices = numDevices;
      kernel = new Kernel<T, NDIM>(std::cout);
      kernel->InitKernel(KEY, VERBOSE, numDevices);
    }

    ~Cuhre()
    {
      delete kernel;
      // QuadDebug(cudaDeviceReset());
    }

    int const BUFSIZE = 512;
#define TAG 0
    template <typename IntegT>
    int
    ExecutePhaseI(IntegT* d_integrand,
                  cuhreResult& res,
                  Volume<T, NDIM>* volume,
                  const int phase1type)
    {
    
      return kernel->IntegrateFirstPhase(d_integrand,
                                         epsrel,
                                         epsabs,
                                         res.estimate,
                                         res.errorest,
                                         res.nregions,
                                         res.nFinishedRegions,
                                         res.neval,
                                         volume);
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

    template <typename IntegT>
    cuhreResult
    integrate(IntegT integrand,
              T epsrel,
              T epsabs,
              Volume<T, NDIM>* volume = nullptr,
              int verbosity = 0,
              int Final = 0,
              int heuristicID = 0,
              int phase1type = 0,
              bool phase2 = false)
    {
      cuhreResult res;

      this->epsrel = epsrel;
      this->epsabs = epsabs;
      kernel->SetFinal(Final);
      kernel->SetVerbosity(verbosity);
      kernel->SetPhase_I_type(phase1type);
      kernel->SetHeuristicID(heuristicID);
      kernel->SetPhase2(phase2);
      

      //cudaMallocManaged((void**)&d_integrand, sizeof(IntegT));
      //memcpy(d_integrand, &integrand, sizeof(IntegT));
      IntegT* d_integrand = quad::cuda_copy_to_managed(integrand);
      CudaCheckError();
      
      kernel->GenerateInitialRegions();
      FIRST_PHASE_MAXREGIONS *= numDevices;

      res.status  = ExecutePhaseI(d_integrand, res, volume, phase1type);
      res.lastPhase = 1;     
      res.status = !(res.errorest <= MaxErr(res.estimate, epsrel, epsabs));
      
      cudaFree(d_integrand);
      return res;
    }
  };
}

#endif
