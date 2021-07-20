#define CATCH_CONFIG_MAIN
#include "cudaPagani/quad/GPUquad/Pagani.cuh"
#include "cudaPagani/quad/quad.h" // for cuhreResult
#include "catch2/catch.hpp"

//#include "fun6.cuh"

// double constexpr integral = 6.371054e-01; // Value is approximate
// double constexpr normalization = 1./integral;

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

template <typename F, int NDIM>
bool
time_and_call(F integrand,
              double epsrel,
              double correct_answer,
              char const* algname)
{
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-40;

  // Why does the integration algorithm need ndim as a template parameter?
  quad::Pagani<double, NDIM> alg;

  auto const t0 = std::chrono::high_resolution_clock::now();
  auto const res = alg.integrate(integrand, epsrel, epsabs);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(res.estimate - correct_answer);
  bool const good = (res.status == 0);
  std::cout << std::scientific << algname << '\t' << epsrel << '\t';
  if (good) {
    std::cout << res.estimate << '\t' << res.errorest << '\t' << absolute_error
              << '\t';
  } else {
    std::cout << "NA\tNA\tNA\t";
  }
  std::cout << res.neval << '\t' << res.nregions << '\t' << dt.count()
            << std::endl;
  return good;
}

TEST_CASE("fun6"){
  SECTION("decreasing epsrel results in non-increasing error estimate"){
    // We start with a very large error tolerance, and will
    // repeatedly decrease the tolerance.

double epsrel = 1.0e-3;
double constexpr epsabs = 1.0e-40;
constexpr int ndim = 6;
double lows[] = {0., 0., 0., 0., 0., 0.};
double highs[] = {1., 1., 1., 1., 1., 1.};

quad::Volume<double, ndim> vol(lows, highs);
quad::Pagani<double, ndim> alg;

double previous_error_estimate = 1.0; // larger than ever should be returned
Fun6 integrand;
while (epsrel > 1.0e-6) {
  printf("About to call alg.integrate\n");
  cuhreResult const res =
    alg.integrate<Fun6>(integrand, epsrel, epsabs, &vol, 0, 1, 0);
  // The integration should have converged.
  bool good = false;

  if (res.status == 0 || res.status == 2) {
    good = true;
  }

  CHECK(good == true);
  std::cout << std::fixed << res.estimate << ",\t" << std::fixed << res.errorest
            << ",\t" << std::fixed << res.nregions << ",\t" << std::fixed
            << res.status << ",\t" << std::endl;
  // The fractional error error estimate should be
  // no larger than the specified fractional error
  // tolerance unless the program does not claim to
  // have confidence in the computed result
  if (good == true)
    CHECK(res.errorest / res.estimate <= epsrel);

  // The error estimate should be no larger than the previous iteration.
  CHECK(res.errorest <= previous_error_estimate);

  // Prepare for the next loop.
  previous_error_estimate = res.errorest;
  epsrel /= 2.0;
}
}
}
;

/*TEST_CASE("genz_1abs_5d")
{
  SECTION("decreasing epsrel results in non-increasing error estimate")
  {
    // We start with a very large error tolerance, and will
    // repeatedly decrease the tolerance.
    double epsrel = 1.0e-3;

    double constexpr epsabs = 1.0e-40;

    double lows[] = {0., 0., 0., 0., 0.};
    double highs[] = {1., 1., 1., 1., 1.};
    constexpr int ndim = 5;
    quad::Volume<double, ndim> vol(lows, highs);
    quad::Cuhre<double, ndim> alg(0, nullptr, 0, 0, 1);

    Genz_1abs_5d integrand;
    double previous_error_estimate = 1.0; // larger than ever should be returned

    while (epsrel > 1.0e-6) {
      cuhreResult const res = alg.integrate(integrand, epsrel, epsabs, &vol);
      // The integration should have converged.
      CHECK(res.status);

      // The fractional error error estimate should be
      // no larger than the specified fractional error
      // tolerance
      CHECK(res.errorest/res.estimate <= epsrel);

      // The error estimate should be no larger than the previous iteration.
      CHECK(res.errorest <= previous_error_estimate);

      // Prepare for the next loop.
      previous_error_estimate = res.errorest;
      epsrel /= 2.0;
    }
  }
};*/
