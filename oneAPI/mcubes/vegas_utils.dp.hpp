#ifndef VEGAS_UTILS_CUH
#define VEGAS_UTILS_CUH

#include <CL/sycl.hpp>
// #include <dpct/dpct.hpp>
#include "oneAPI/mcubes/seqCodesDefs.hh"

#define BLOCK_DIM_X 128
#define RAND_MAX 2147483647

#include <cmath>

class Custom_generator {
  const uint32_t a = 1103515245;
  const uint32_t c = 12345;
  const uint32_t one = 1;
  const uint32_t expi = 31;
  uint32_t p = one << expi;
  uint32_t custom_seed = 0;
  uint64_t temp = 0;

public:
  Custom_generator(uint32_t seed) : custom_seed(seed){};

  double
  operator()()
  {
    temp = a * custom_seed + c;
    custom_seed = temp & (p - 1);
    // return (double)custom_seed / (double)p;
    return static_cast<double>(custom_seed) / static_cast<double>(p);
  }

  void
  SetSeed(uint32_t seed)
  {
    custom_seed = seed;
  }
};

class Curand_generator {
public:
  Curand_generator(sycl::nd_item<3> item_ct1) {}

  Curand_generator(unsigned int seed, sycl::nd_item<3> item_ct1) {}

  double
  operator()()
  {

    return 0.;
  }
};

namespace mcubes {
  template <typename T, typename U>
  struct TypeChecker {
    // static const bool value = false;
    static constexpr bool
    is_custom_generator()
    {
      return false;
    }
  };

  template <typename Custom_generator>
  struct TypeChecker<Custom_generator, Custom_generator> {
    // static const bool value = true;

    static constexpr bool
    is_custom_generator()
    {
      return true;
    }
  };
}

class Internal_Vegas_Params {
  static constexpr int ndmx = 500;
  static constexpr int mxdim = 20;
  static constexpr double alph = 1.5;

public:
  static constexpr int
  get_NDMX()
  {
    return ndmx;
  }

  static constexpr int
  get_NDMX_p1()
  {
    return ndmx + 1;
  }

  static constexpr double
  get_ALPH()
  {
    return alph;
  }

  static constexpr int
  get_MXDIM()
  {
    return mxdim;
  }

  constexpr static int
  get_MXDIM_p1()
  {
    return mxdim + 1;
  }
};

__inline__ double
ComputeNcubes(double ncall, int ndim)
{
  double ncubes = 1.;
  double intervals_per_dim = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim);
  for (int dim = 1; dim <= ndim; dim++) {
    ncubes *= intervals_per_dim;
  }

  return ncubes;
}

__inline__ int
Compute_samples_per_cube(double ncall, double ncubes)
{
  int npg = IMAX(ncall / ncubes, 2);
  return npg;
}

struct Kernel_Params {
  double ncubes = 0.;
  int npg = 0;
  uint32_t nBlocks = 0;
  uint32_t nThreads = 0;
  uint32_t totalNumThreads = 0;
  uint32_t totalCubes = 0;
  int extra = 0;
  int LastChunk = 0; // how many chunks for the last thread

  Kernel_Params(double ncall, int chunkSize, int ndim)
  {
    ncubes = ComputeNcubes(ncall, ndim);
    npg = Compute_samples_per_cube(ncall, ncubes);

    totalNumThreads = (uint32_t)((ncubes) / chunkSize);
    totalCubes = totalNumThreads * chunkSize;
    extra = totalCubes - ncubes;
    LastChunk = chunkSize - extra;
    nBlocks = totalNumThreads % BLOCK_DIM_X == 0 ?
                totalNumThreads / BLOCK_DIM_X :
                totalNumThreads / BLOCK_DIM_X + 1;
    nThreads = BLOCK_DIM_X;
  }
};

__inline__ bool
PrecisionAchieved(double estimate,
                  double errorest,
                  double epsrel,
                  double epsabs)
{
  if (std::abs(errorest / estimate) <= epsrel || errorest <= epsabs) {
    return true;
  } else
    return false;
}

__inline__ int
GetStatus(double estimate,
          double errorest,
          int iteration,
          double epsrel,
          double epsabs)
{
  if (PrecisionAchieved(estimate, errorest, epsrel, epsabs) && iteration >= 5) {
    return 0;
  } else
    return 1;
}

__inline__ int
GetChunkSize(const double ncall)
{
  double small = 1.e7;
  double large = 8.e9;

  if (ncall <= small)
    return 32;
  else if (ncall <= large)
    return 2048;
  else
    return 4096;
}

/*
  returns true if it can update params for an extended run, updates two params
  returns false if it has increased both params to their maximum allowed values
  this maximum is not configurable by the user, placeholder values are currently
  placed
 */

bool
CanAdjustNcallOrIters(double ncall, int totalIters)
{
  if (ncall >= 8.e9 && totalIters >= 100)
    return false;
  else
    return true;
}

bool
AdjustParams(double& ncall, int& totalIters)
{
  if (ncall >= 8.e9 && totalIters >= 100) {
    // printf("Adjusting will return false\n");
    return false;
  } else if (ncall >= 8.e9) {
    //  printf("Adjusting will increase iters by 10 current value:%i\n",
    //  totalIters);
    totalIters += 10;
    return true;
  } else if (ncall >= 1.e9) {
    // printf("Adjusting will increase ncall by 1e9 current value:%e\n", ncall);
    ncall += 1.e9;
    return true;
  } else {
    //  printf("Adjusting will multiply ncall by 10 current value:%e\n", ncall);
    ncall *= 10.;
    return true;
  }
}

#endif
