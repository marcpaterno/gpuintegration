#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include "Interp2D.h"
#include <array>
#include <math.h> 

double Evaluate(quad::Interp2D f, double inputX, double inputY){
    ViewVectorDouble results("results", 1);
    ViewVectorDouble::HostMirror hostResults = Kokkos::create_mirror_view(results);
        
    size_t numInterpolations = 1;
    Kokkos::parallel_for(
        "Copy_from_stdArray", 
        numInterpolations, [=] __device__ (const int64_t index) {
            results(0) = f(inputX, inputY);
        });
    Kokkos::deep_copy(hostResults, results);
    return hostResults(0);
}

double Clamp(quad::Interp2D f, double inputX, double inputY){
    ViewVectorDouble results("results", 1);
    ViewVectorDouble::HostMirror hostResults = Kokkos::create_mirror_view(results);
        
    size_t numInterpolations = 1;
    Kokkos::parallel_for(
        "Copy_from_stdArray", 
        numInterpolations, [=] __device__ (const int64_t index) {
            results(0) = f.clamp(inputX, inputY);
        });
        
    Kokkos::deep_copy(hostResults, results);
    return hostResults(0);
}

TEST_CASE("clamp interface works"){
    constexpr std::size_t nx = 3; // rows
    constexpr std::size_t ny = 2; // cols
    std::array<double, nx> xs = {1., 2., 3.};
    std::array<double, ny> ys = {4., 5.};
    std::array<double, ny * nx> zs;
    
    
    auto fxy = [](double x, double y) { return 3 * x * y + 2 * x + 4 * y; };

    for (std::size_t i = 0; i != nx; ++i) {
        double x = xs[i];
        for (std::size_t j = 0; j != ny; ++j) {
            double y = ys[j];
            zs[j * nx + i] = fxy(x, y);
        }
    }
    
    quad::Interp2D f(xs, ys, zs);
    
    SECTION("interpolation works")
    {
        double x = 2.5;
        double y = 4.5;
        double true_result = 56.75;
        
        double hostRes = Evaluate(f, x, y); 
        CHECK(hostRes == true_result);
    }
    
    SECTION("extrapolation gets clamped")
    {
        double clampRes = Clamp(f, 0., 4.5);
        double interpResult = Clamp(f, 1., 4.5);
        CHECK(clampRes == interpResult); // to the left

        clampRes = Clamp(f, 4., 4.5);
        interpResult = Clamp(f, 3., 4.5);
        CHECK(clampRes == interpResult);

        clampRes = Clamp(f, 2., 3.);
        interpResult = Clamp(f, 2., 4.);
        CHECK(clampRes == interpResult);

        clampRes = Clamp(f, 2., 5.5);
        interpResult = Clamp(f, 2., 5.);
        CHECK(clampRes == interpResult);

        clampRes = Clamp(f, 0., 0.);
        interpResult = Clamp(f, 1., 4.);
        CHECK(clampRes == interpResult);

        clampRes = Clamp(f, 4., 3.);
        interpResult = Clamp(f, 3., 4.);
        CHECK(clampRes == interpResult);

        clampRes = Clamp(f, 0., 6.);
        interpResult = Clamp(f, 1., 5.);
        CHECK(clampRes == interpResult);

        clampRes = Clamp(f, 4., 6.);
        interpResult = Clamp(f, 3., 5.);
        CHECK(clampRes == interpResult);
    }
}

TEST_CASE("Interp2D exact at knots")
{
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 2;
  std::array<double, nx> const xs = {1., 2., 3.};
  std::array<double, ny> const ys = {4., 5.};
  auto fxy = [](double x, double y) { return 3 * x * y + 2 * x + 4 * y; };
  std::array<double, ny * nx> zs;

  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys[j];
      zs[j * nx + i] = fxy(x, y);
    }
  }

  quad::Interp2D f(xs, ys, zs);
    for (std::size_t i = 0; i != nx; ++i) {
        double x = xs[i];
        for (std::size_t j = 0; j != ny; ++j) {
            double y = ys[j];
            CHECK(zs[j * nx + i] == fxy(x, y));
            double interpResult = Evaluate(f, x, y);
            CHECK(zs[j * nx + i] == interpResult);
        }
    }
}

TEST_CASE("Interp2D on bilinear")
{
  constexpr std::size_t nx = 3;
  constexpr std::size_t ny = 4;
  std::array<double, nx> const xs = {1., 2., 3.};
  std::array<double, ny> const ys = {1., 2., 3., 4.};
  std::array<double, ny * nx> zs;

  auto fxy = [](double x, double y) { return 2 * x + 3 * y - 5; };

  for (std::size_t i = 0; i != nx; ++i) {
    double x = xs[i];
    for (std::size_t j = 0; j != ny; ++j) {
      double y = ys[j];
      zs[j * nx + i] = fxy(x, y);
      CHECK(zs[j * nx + i] == fxy(x, y));
    }
  }

  quad::Interp2D f(xs, ys, zs);

  double interpResult = Evaluate(f, 2.5, 1.5);
  CHECK(interpResult == 4.5);
}

int main( int argc, char* argv[] ) {
  int result = 0;
  Kokkos::initialize();
  {  
    result = Catch::Session().run( argc, argv );
  }
  Kokkos::finalize();  
  return result;
}