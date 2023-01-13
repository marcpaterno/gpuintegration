#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "test/cuda/pagani/model.cuh"
#include "test/cuda/pagani/model.hh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

TEST_CASE("Toy Integrand Yields Same results on cpu and gu")
{
  std::cout << "In executable\n";
  size_t numEvaluations = 10;
  std::vector<double> cpu_results = cpuExecute();
  std::vector<double> gpu_results = gpuExecute();

  for (size_t i = 0; i < numEvaluations; ++i) {
    CHECK(cpu_results[i] == Approx(gpu_results[i]).epsilon(1.e-12));
    // printf("%.15e vs %.15e\n", cpu_results[i], gpu_results[i]);
  }
}