#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include "kokkos/kokkosPagani/quad/Interp1D.h"
//#include "quad.h"
#include <array>
#include <math.h>

typedef Kokkos::View<quad::Interp1D*, Kokkos::CudaUVMSpace> ViewVectorInterp1D;

TEST_CASE("Initialization from std::array")
{
  const size_t s = 9;
  std::array<double, s> xs = {1., 2., 3., 4., 5., 6, 7., 8., 9.};
  std::array<double, s> ys = {2., 4., 8., 16., 32., 64., 128., 256., 512.};

  quad::Interp1D interpolator(xs, ys);
  ViewVectorInterp1D object("Interp1D view", 1);
  object(0) = interpolator;

  SECTION("Values propertly set on the device")
  {
    double sum_x = 0.;
    double sum_y = 0.;

    Kokkos::parallel_reduce(
      "Reduce",
      s,
      KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
        valueToUpdate += object(0).interpC(index);
      },
      sum_x);

    Kokkos::parallel_reduce(
      "Reduce",
      s,
      KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
        valueToUpdate += object(0).interpT(index);
      },
      sum_y);

    CHECK(sum_x == 45.);
    CHECK(sum_y == 1022.);
  }
}

TEST_CASE("Initialization from C-style array")
{
  const size_t s = 9;
  double xs[9] = {1., 2., 3., 4., 5., 6, 7., 8., 9.};
  double ys[9] = {2., 4., 8., 16., 32., 64., 128., 256., 512.};

  quad::Interp1D interpolator(xs, ys, s);
  ViewVectorInterp1D object("Interp1D view", 1);
  object(0) = interpolator;

  SECTION("Values propertly set on the device")
  {
    double sum_x = 0.;
    double sum_y = 0.;
    Kokkos::parallel_reduce(
      "Reduce",
      s,
      KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
        valueToUpdate += object(0).interpC(index);
      },
      sum_x);

    Kokkos::parallel_reduce(
      "Reduce",
      s,
      KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
        valueToUpdate += object(0).interpT(index);
      },
      sum_y);

    CHECK(sum_x == 45.);
    CHECK(sum_y == 1022.);
  }
}

TEST_CASE("Initialization from Kokkos View")
{
  const size_t s = 9;
  HostVectorDouble xs("xs", s);
  HostVectorDouble ys("ys", s);

  for (size_t i = 0; i < s; ++i) {
    xs[i] = i + 1;
    ys[i] = pow(2, i + 1);
  }

  quad::Interp1D interpolator(xs, ys);
  ViewVectorInterp1D object("Interp1D view", 1);
  object(0) = interpolator;

  double sum_x = 0.;
  double sum_y = 0.;
  Kokkos::parallel_reduce(
    "Reduce",
    s,
    KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
      valueToUpdate += object(0).interpC(index);
    },
    sum_x);

  Kokkos::parallel_reduce(
    "Reduce",
    s,
    KOKKOS_LAMBDA(const int64_t index, double& valueToUpdate) {
      valueToUpdate += object(0).interpT(index);
    },
    sum_y);

  CHECK(sum_x == 45.);
  CHECK(sum_y == 1022.);
}

TEST_CASE("Interp1D exact at knots")
{

  // user's input arrays
  const size_t s = 9;
  std::array<double, s> xs = {1., 2., 3., 4., 5., 6, 7., 8., 9.};
  std::array<double, s> ys = xs;

  auto Transform = [](std::array<double, s>& ys) {
    for (double& elem : ys)
      elem = 2 * elem * (3 - elem) * std::cos(elem);
  };
  Transform(ys);

  // instantiate interpolator, it's views reside on the device
  quad::Interp1D interpolator(xs, ys);
  ViewVectorInterp1D object("Interp1D view", 1);
  object(0) = interpolator;

  // prepare the values that we will interpolate on
  ViewVectorDouble input("input", s);
  ViewVectorDouble::HostMirror hostInput = Kokkos::create_mirror_view(input);

  for (size_t i = 0; i < s; ++i) {
    hostInput(i) = xs[i];
  }
  Kokkos::deep_copy(input, hostInput);

  // view to store results and view back at cpu
  ViewVectorDouble results("results", s);
  ViewVectorDouble::HostMirror hostResults =
    Kokkos::create_mirror_view(results);

  uint32_t nBlocks = 1;
  uint32_t nThreads = 1;
  Kokkos::TeamPolicy<> mainKernelPolicy(nBlocks, nThreads);

  Kokkos::parallel_for(
    "Phase1", mainKernelPolicy, [=] __device__(const member_type team_member) {
      for (size_t i = 0; i < s; i++)
        results(i) = object(0)(input(i));
    });

  Kokkos::deep_copy(hostResults, results);
  for (std::size_t i = 0; i < s; ++i) {
    CHECK(ys[i] == hostResults(i));
  }
}

TEST_CASE("Interp1D on quadratic")
{
  const size_t s = 3;
  std::array<double, s> xs = {1., 2., 3.};
  std::array<double, s> ys = xs;

  auto Transform = [](std::array<double, s>& ys) {
    for (auto& elem : ys)
      elem = elem * elem;
  };
  Transform(ys);
  quad::Interp1D interpolator(xs, ys);
  ViewVectorInterp1D object("Interp1D view", 1);
  object(0) = interpolator;

  ViewVectorDouble results("results", 1);
  ViewVectorDouble::HostMirror hostResults =
    Kokkos::create_mirror_view(results);

  double input = 1.41421;
  uint32_t nBlocks = 1;
  uint32_t nThreads = 1;
  Kokkos::TeamPolicy<> mainKernelPolicy(nBlocks, nThreads);

  Kokkos::parallel_for(
    "Phase1", mainKernelPolicy, [=] __device__(const member_type team_member) {
      for (size_t i = 0; i < s; i++)
        results(i) = object(0)(input);
    });

  Kokkos::deep_copy(hostResults, results);
  CHECK(hostResults(0) == Approx(2.24263).epsilon(1e-4));
}

int
main(int argc, char* argv[])
{
  int result = 0;
  Kokkos::initialize();
  {
    result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return result;
}