#ifndef CUDACUHRE_QUAD_UTIL_CUDA_UTIL_H
#define CUDACUHRE_QUAD_UTIL_CUDA_UTIL_H

#include <CL/sycl.hpp>
#include "oneAPI/pagani/quad/quad.h"
#include "common/oneAPI/cudaDebugUtil.h"

#include <float.h>
#include <stdio.h>

#include "oneAPI/pagani/quad/deviceProp.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void
warmUpKernel()
{}

#define Zap(d) memset(d, 0, sizeof(d))

inline double
MaxErr(double avg, double epsrel, double epsabs)
{
  return sycl::max(epsrel * sycl::fabs(avg), epsabs);
}

template <size_t ndim>
constexpr size_t
CuhreFuncEvalsPerRegion()
{
  return (1 + 2 * ndim + 2 * ndim + 2 * ndim + 2 * ndim +
          2 * ndim * (ndim - 1) + 4 * ndim * (ndim - 1) +
          4 * ndim * (ndim - 1) * (ndim - 2) / 3 + (1 << ndim));
}

#endif
