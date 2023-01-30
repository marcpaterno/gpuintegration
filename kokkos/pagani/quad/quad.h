#ifndef KOKKOSCUHRE_QUAD_H
#define KOKKOSCUHRE_QUAD_H

#define BLOCK_SIZE 64
#include <Kokkos_Core.hpp>

// headers from cudaUtil.h
// #include "cudaDebugUtil.h""

#include "common/kokkos/cudaMemoryUtil.h"
#include <cmath>
#include <float.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>

#define INFTY DBL_MAX
#define Zap(d) memset(d, 0, sizeof(d))

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

typedef Kokkos::View<GlobalBounds*,
                     Kokkos::DefaultExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
  ScratchViewGlobalBounds;

//-------------------------------------------------------------------------------
// Device Views

template <typename T>
struct Structures {
  constViewVectorDouble _gpuG;
  constViewVectorDouble _cRuleWt;
  constViewVectorDouble _GPUScale;
  constViewVectorDouble _GPUNorm;
  constViewVectorInt _gpuGenPos;
  constViewVectorInt _gpuGenPermGIndex;
  constViewVectorInt _gpuGenPermVarCount;
  constViewVectorInt _gpuGenPermVarStart;
  ViewVectorSize_t _cGeneratorCount;
};

typedef Kokkos::View<Structures<double>*, Kokkos::Cuda> ViewStructures;

#define NRULES 5

inline __device__ __host__ double
MaxErr(double avg, double epsrel, double epsabs)
{
  return max(epsrel * std::abs(avg), epsabs);
}

template <typename T, int NDIM>
struct Volume {

  T lows[NDIM] = {0.0};
  T highs[NDIM];

  __host__
  Volume()
  {
    for (T& x : highs)
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
