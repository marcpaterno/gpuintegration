#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include "cuda/pagani/quad/util/cudaArray.cuh"

namespace quad {

  template <size_t Order>
  class polynomial {
  private:
    const gpu::cudaArray<double, Order> coeffs;

  public:
    __host__ __device__
    polynomial(gpu::cudaArray<double, Order> coeffs)
      : coeffs(coeffs)
    {}

    __host__ __device__ constexpr double
    operator()(const double x) const
    {
      double out = 0.0;
      for (auto i = 0u; i < Order; i++)
        out = coeffs[i] + x * out;
      return out;
    }
  };
}
#endif