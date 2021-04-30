#ifndef KOKKOSCUHRE_QUAD_H
#define KOKKOSCUHRE_QUAD_H

#define BLOCK_SIZE 256
#include <Kokkos_Core.hpp>

//headers from cudaUtil.h
//#include "cudaDebugUtil.h""
#include <float.h>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <string>

#define INFTY DBL_MAX
#define Zap(d) memset(d, 0, sizeof(d))

//if Kokkos::Cuda tests pass, if Kokkos::CudaSpace tests fail
typedef Kokkos::View<int*, Kokkos::CudaSpace>   ViewVectorInt;
typedef Kokkos::View<float*, Kokkos::CudaSpace>   ViewVectorFloat;
typedef Kokkos::View<double*, Kokkos::CudaSpace>   ViewVectorDouble;
typedef Kokkos::View<size_t*, Kokkos::CudaSpace>   ViewVectorSize_t;

typedef Kokkos::View<const double*, Kokkos::CudaSpace>   constViewVectorDouble;
typedef Kokkos::View<const int*, Kokkos::CudaSpace>   constViewVectorInt;
typedef Kokkos::View<const size_t*, Kokkos::CudaSpace>   constViewVectorSize_t;

typedef Kokkos::TeamPolicy<>    team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;
typedef Kokkos::TeamPolicy<>    team_policy;
typedef Kokkos::TeamPolicy<>::member_type  member_type;
typedef Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewDouble;    
typedef Kokkos::View<int*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewInt; 
typedef Kokkos::View<int*, Kokkos::Serial> HostVectorInt;
typedef Kokkos::View<double*, Kokkos::Serial> HostVectorDouble;
typedef Kokkos::View<size_t*, Kokkos::Serial> HostVectorSize_t;

struct Bounds {
  double lower, upper;
};

struct Result {
  double avg, err;
  int bisectdim;
};

template <int dim>
struct Region {
  int div;
  Result result;
  Bounds bounds[dim];
};

struct GlobalBounds {
  double unScaledLower, unScaledUpper;
};

typedef Kokkos::View<GlobalBounds*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewGlobalBounds;

struct cuhreResult {

  cuhreResult()
  {
    estimate = 0.;
    errorest = 0.;
    neval = 0.;
    nregions = 0.;
    status = 1.;
    //activeRegions = 0.;
    phase2_failedblocks = 0.;
	lastPhase = 0;
    nFinishedRegions = 0;
  };
	
  double estimate;
  double errorest;
  size_t neval;
  size_t nregions;
  size_t nFinishedRegions;
  int status;
  int lastPhase;
  //size_t activeRegions;    // is not currently being set
  size_t phase2_failedblocks; // is not currently being set
};

template <typename T>
struct Structures {
  //__host__ __device__
  //Structures(){}

 // ~Structures() {}

  /*const*/ViewVectorDouble _gpuG;
  /*const*/ViewVectorDouble _cRuleWt;
  /*const*/ViewVectorDouble _GPUScale;
  /*const*/ViewVectorDouble _GPUNorm;
  /*const*/ViewVectorInt _gpuGenPos;
  /*const*/ViewVectorInt  _gpuGenPermGIndex;
  /*const*/ViewVectorInt _gpuGenPermVarCount;
  /*const*/ViewVectorInt _gpuGenPermVarStart;
  ViewVectorSize_t _cGeneratorCount;
};

typedef Kokkos::View<Structures<double>*, Kokkos::Cuda>   ViewStructures;

#define NRULES 5

inline
__device__ __host__
double MaxErr(double avg, double epsrel, double epsabs) {
  return max(epsrel * std::abs(avg), epsabs);
}

template <typename T, int NDIM>
struct Volume {

    T lows[NDIM]  = {0.0};
    T highs[NDIM];

    __host__
    Volume()
    {
		for(T& x: highs)
			x = 1.0;
		
	}

    __host__ __device__
    Volume(T const* l, T const* h)
    {
      std::memcpy(lows, l, NDIM * sizeof(T));
      std::memcpy(highs, h, NDIM * sizeof(T));
    }

};

#endif