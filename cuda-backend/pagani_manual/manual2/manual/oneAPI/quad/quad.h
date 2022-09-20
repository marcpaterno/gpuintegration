#ifndef ONE_API_QUAD_QUAD_h
#define  ONE_API_QUAD_QUAD_h

#define TIMING_DEBUG 1
#define BLOCK_SIZE 64



#include <fstream>
#include <string>
#include <vector>
#include "cuda/cudaPagani/quad/util/cuhreResult.cuh"

//double errcoeff[3] = {5, 1, 5};

#define NRULES 5


struct Result {
  double avg, err;
  int bisectdim;
};

struct Bounds {
  double lower, upper;
};

struct GlobalBounds {
  double unScaledLower, unScaledUpper;
};

template <int dim>
struct Region {
  int div;
  Result result;
  Bounds bounds[dim];
};


#endif
