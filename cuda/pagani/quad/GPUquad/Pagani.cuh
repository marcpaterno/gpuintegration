#ifndef CUDACUHRE_QUAD_GPUQUAD_CUHRE_CUH
#define CUDACUHRE_QUAD_GPUQUAD_CUHRE_CUH

#include "cuda/pagani/quad/util/cudaMemoryUtil.h"
#include "cuda/pagani/quad/util/cudaTimerUtil.h"

#include "cuda/pagani/quad/GPUquad/Kernel.cuh"
#include "cuda/pagani/quad/util/Volume.cuh"

namespace quad {

  template <typename T, int NDIM>
  class Pagani {

    int key;

    int verbose;
    int numDevices;

    T epsrel;
    T epsabs;
    Kernel<T, NDIM> kernel;
    std::ofstream log;

  public:
    // Note that this also acts as the default constructor.
    explicit Pagani(int key = 0, int verbose = 0, int numDevices = 1)
      : key(key)
      , verbose(verbose)
      , numDevices(numDevices)
      , epsrel(0.0)
      , epsabs(0.0)
      , kernel(std::cout)
    {
      kernel.InitKernel(key, verbose, numDevices);
    }

    Pagani(Pagani const&) = delete;            // no copying
    Pagani(Pagani&&) = delete;                 // no move copy
    Pagani& operator=(Pagani const&) = delete; // no assignment
    Pagani& operator=(Pagani&&) = delete;      // no move assignment


    template <typename IntegT>
    int
    ExecutePhaseI(IntegT* d_integrand,
                  cuhreResult<T>& res,
                  Volume<T, NDIM> const* volume)
    {

      return kernel.IntegrateFirstPhase(d_integrand,
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
    VerboseResults
    EvaluateAtCuhrePoints(IntegT integrand, Volume<T, NDIM>* volume = nullptr)
    {
      IntegT* d_integrand = quad::cuda_copy_to_managed(integrand);
      CudaCheckError();
      kernel.GenerateInitialRegions();
      VerboseResults resultsObj;
      resultsObj.NDIM = NDIM;
      kernel.EvaluateAtCuhrePoints(d_integrand, resultsObj, volume);

      return resultsObj;
    }

    template <typename IntegT>
    cuhreResult<T>
    integrate(IntegT& integrand,
              T epsrel,
              T epsabs,
              Volume<T, NDIM> const* volume = nullptr,
              int verbosity = 0,
              int Final = 0,
              int heuristicID = 0,
              int phase1type = 0)
    {
      cuhreResult<T> res;

      this->epsrel = epsrel;
      this->epsabs = epsabs;
      kernel.SetVerbosity(verbosity);
      kernel.SetHeuristicID(heuristicID);

      IntegT* d_integrand = quad::cuda_copy_to_managed(integrand);
      CudaCheckError();

      kernel.GenerateInitialRegions();
      FIRST_PHASE_MAXREGIONS *= numDevices;

      res.status = ExecutePhaseI(d_integrand, res, volume);
      res.lastPhase = 1;
      res.status = !(res.errorest <= MaxErr(res.estimate, epsrel, epsabs));
      d_integrand->~IntegT();
      cudaFree(d_integrand);
      CudaCheckError();
      return res;
    }
  };
}

#endif
