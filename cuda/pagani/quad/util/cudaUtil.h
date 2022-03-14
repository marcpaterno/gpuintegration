#ifndef CUDACUHRE_QUAD_UTIL_CUDA_UTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDA_UTIL_H

#include "cuda/pagani/quad/quad.h"
#include "cuda/pagani/quad/util/cudaDebugUtil.h"

#include <float.h>
#include <stdio.h>

#include "cuda/pagani/quad/deviceProp.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void
warmUpKernel()
{}

#define INFTY DBL_MAX
#define Zap(d) memset(d, 0, sizeof(d))

inline __device__ __host__ double
MaxErr(double avg, double epsrel, double epsabs)
{
  return max(epsrel * std::abs(avg), epsabs);
}

#endif
