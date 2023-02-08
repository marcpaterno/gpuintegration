#include "catch2/catch.hpp"
#include "kokkos/pagani/quad/GPUquad/Workspace.cuh"
#include "common/integration_result.hh"

struct Fun6 {

  KOKKOS_INLINE_FUNCTION double
  operator()(double u, double v, double w, double x, double y, double z)
  {
    return (12.0 / (7.0 - 6 * log(2.0) * log(2.0) + log(64.0))) *
           (u * v + (pow(w, y) * x * y) / (1 + u) + z * z);
  }
};

TEST_CASE("Accuracy Improves on Smaller Relative Error Tolerances")
{
  // We start with a very large error tolerance, and will
  // repeatedly decrease the tolerance.

  double epsrel = 1.0e-3;
  double constexpr epsabs = 1.0e-40;
  constexpr int ndim = 6;
  constexpr bool use_custom = true;
  constexpr bool collect_iters = false;
  bool constexpr predict_split = false;
  int constexpr debug_level = 0;
  quad::Volume<double, ndim> vol;
  Workspace<double, ndim, use_custom> alg;

  double previous_error_estimate = 1.0; // larger than ever should be returned
  Fun6 integrand;
  while (epsrel > 1.0e-6) {
    auto const res =
      alg.integrate<Fun6, collect_iters, predict_split, debug_level>(
        integrand, epsrel, epsabs, vol);

    CHECK(res.status == 0);

    if (res.status == true)
      CHECK(res.errorest / res.estimate <= epsrel);

    // The error estimate should be no larger than the previous iteration.
    CHECK(res.errorest <= previous_error_estimate);

    // Prepare for the next loop.
    previous_error_estimate = res.errorest;
    epsrel /= 2.0;
  }
}
