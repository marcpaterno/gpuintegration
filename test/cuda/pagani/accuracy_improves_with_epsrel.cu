#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "cuda/pagani/quad/GPUquad/Workspace.cuh"
#include "common/integration_result.hh"

static double const fun6_normalization =
  12.0 / (7.0 - 6 * log(2.0) * std::log(2.0) + log(64.0));

double
__fun6(double u, double v, double w, double x, double y, double z)
{
  return fun6_normalization *
         (u * v + (std::pow(w, y) * x * y) / (1 + u) + z * z);
}

struct Fun6 {

  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z)
  {
    return (12.0 / (7.0 - 6 * log(2.0) * log(2.0) + log(64.0))) *
           (u * v + (pow(w, y) * x * y) / (1 + u) + z * z);
  }
};

struct Genz_1abs_5d {

  __device__ __host__ Genz_1abs_5d(){};

  __device__ __host__ double
  operator()(double v, double w, double x, double y, double z)
  {
    return (1. / 6.371054e-01) *
           fabs(cos(4. * v + 5. * w + 6. * x + 7. * y + 8. * z));
  }
};


TEST_CASE("Accuracy Improves on Smaller Relative Error Tolerances"){
    // We start with a very large error tolerance, and will
    // repeatedly decrease the tolerance.

    double epsrel = 1.0e-3;
	double constexpr epsabs = 1.0e-40;
	constexpr int ndim = 6;

	quad::Volume<double, ndim> vol;
	Workspace<double, ndim> alg;

	double previous_error_estimate = 1.0; // larger than ever should be returned
	Fun6 integrand;
	while (epsrel > 1.0e-6) {
		auto const res = alg.integrate<Fun6>(integrand, epsrel, epsabs, vol);

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

