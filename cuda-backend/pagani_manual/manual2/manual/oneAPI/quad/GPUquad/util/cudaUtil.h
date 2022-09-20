#ifndef ONE_API_QUAD_UTIL_CUDA_UTIL_H
#define ONE_API_QUAD_UTIL_CUDA_UTIL_H


#include <float.h>
#include <stdio.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


#define INFTY DBL_MAX
#define Zap(d) memset(d, 0, sizeof(d))

inline double
MaxErr(double avg, double epsrel, double epsabs)
{
  return max(epsrel * std::abs(avg), epsabs);
}

#endif
