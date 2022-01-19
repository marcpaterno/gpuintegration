#define CATCH_CONFIG_RUNNER
#include "Cuhre.cuh"
#include "catch.hpp"
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// Kokkos doesn't offer an API for quering device memory usage
// Must test that CUDA's API can be utilized properly

TEST_CASE("Appropriate Decrease in Monitored Available GPU Memory")
{
  typedef Kokkos::View<double*, Kokkos::CudaSpace> ViewDouble;

  size_t initial_free_mem = GetAmountFreeMem();
  size_t totalMem1 = GetTotalMem();
  Cuhre<double, 2> cuhre;
  size_t totalMem2 = GetTotalMem();

  SECTION("Total Mem Constant") { CHECK(totalMem2 == totalMem1); }

  size_t numDoublesToFit = initial_free_mem / sizeof(double);

  ViewDouble dummyArray("dummy", numDoublesToFit / 2);
  size_t AfterFirstDummy = GetAmountFreeMem();

  SECTION("Impact of Dummy Array Allocation Visible")
  {
    CHECK(AfterFirstDummy < initial_free_mem);
  }

  size_t AfterFirstDummy2 = GetAmountFreeMem();
  SECTION("Free Mem constant unless additional allcoation")
  {
    CHECK(AfterFirstDummy2 == AfterFirstDummy);
  }

  ViewDouble dummyArray2("dummy2", numDoublesToFit / 4);
  size_t AfterSecondDummy = GetAmountFreeMem();
  SECTION("Additional Allocation yields less free memory")
  {
    CHECK(AfterSecondDummy < AfterFirstDummy);
  }

  ViewDouble dummyArray3("dummy3", numDoublesToFit / 8);

  size_t AfterThirdDummy = GetAmountFreeMem();
  CHECK(AfterThirdDummy < AfterSecondDummy);

  ViewDouble dummyArray4("dummy3", 10000000);
  printf("Free mem:%zu, %zu, %zu, %zu\n",
         GetAmountFreeMem(),
         AfterThirdDummy,
         AfterSecondDummy,
         AfterFirstDummy);
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