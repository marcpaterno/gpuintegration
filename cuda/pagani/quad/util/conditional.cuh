#ifndef CONDITIONAL_H
#define CONDITIONAL_H

#include "cuda/pagani/quad/GPUquad/Interp1D.cuh"
#include "cuda/pagani/quad/GPUquad/Interp2D.cuh"
#include "cuda/pagani/quad/GPUquad/Polynomial.cuh"

struct GPU {
  template <size_t order>
  using polynomial = quad::polynomial<order>;
  typedef quad::Interp2D Interp2D;
  typedef quad::Interp1D Interp1D;
  // typedef quad::ez ez;
};

struct CPU {
  // commented out temporarily to keep y3_cluster independent from PAGANI
  // template<size_t order>
  // using polynomial = y3_cluster::polynomial<order>;
  // typedef y3_cluster::Interp2D Interp2D;
  // typedef y3_cluster::Interp1D Interp1D;
};

#endif