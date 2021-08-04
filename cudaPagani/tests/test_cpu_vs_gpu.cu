#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "cudaPagani/tests/model.cuh"
#include "cudaPagani/tests/model.hh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>


TEST_CASE("Toy Integrand Yields Same results on cpu and gu"){
    std::cout<<"In executable\n";
    size_t numEvaluations = 10;
    double* cpu_results = cpuExecute();
    double* gpu_results = gpuExecute();
    
    for(size_t i = 0; i <numEvaluations; ++i)
        CHECK(cpu_results[i] == gpu_results[i]);
}