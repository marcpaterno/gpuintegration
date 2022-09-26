#ifndef CUDA_FUNC_EVAL
#define CUDA_FUNC_EVAL

#include "cuda/pagani/quad/util/cudaArray.cuh"

namespace quad {

  template <size_t ndim>
  class Feval {
  public:
    double point[ndim] = {0.};

    GlobalBounds global_bounds[ndim];
    Bounds region_bounds[ndim];
    double feval = 0.;
    size_t feval_index;

    Feval() {}

    Feval(const Feval& other)
    {

      for (int dim = 0; dim < ndim; ++dim) {
        point[dim] = other.point[dim];
        global_bounds[dim] = other.global_bounds[dim];
        region_bounds[dim] = other.region_bounds[dim];
      }
      feval_index = other.feval_index;
      feval = other.feval;
    }

    __host__ __device__ void
    store(gpu::cudaArray<double, ndim> x,
          GlobalBounds globalBounds[],
          Bounds sub_region[])
    {
      for (size_t dim = 0; dim < ndim; ++dim) {
        point[dim] = x[dim];
        global_bounds[dim] = globalBounds[dim];
        region_bounds[dim] = sub_region[dim];
      }
    }

    __host__ __device__ void
    store(double res, size_t feval_id)
    {
      feval = res;
      feval_index = feval_id;
    }
  };

  template <size_t ndim>
  class Func_Evals {
  public:
    // put allocation of funct_eval here, and we will just create the object
    const size_t num_fevals = pagani::CuhreFuncEvalsPerRegion<ndim>();
    Feval<ndim>* fevals_list = nullptr;

    __host__ __device__ quad::Feval<ndim>&
    operator[](std::size_t i)
    {
      return fevals_list[i];
    }
  };

}

#endif