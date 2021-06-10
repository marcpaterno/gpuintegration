#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "integrands/hmf_t.cuh"
#include "quad/util/conditional.cuh"
#include <chrono>
#include <fstream>
#include <iostream>

template <class M>
M
make_from_file(char const* filename)
{
  static_assert(std::is_default_constructible<M>::value,
                "Type must be default constructable");
  char const* basedir = std::getenv("PAGANI_DIR");
  std::string fname(basedir);
  fname += "/tests/";
  fname += filename;
  std::cout << "Filename:" << fname << std::endl;
  std::ifstream in(fname);
  if (!in) {
    std::string msg("Failed to open file: ");
    msg += fname;
    throw std::runtime_error(msg);
  }

  M result;
  in >> result;
  return result;
}

template <typename IntegT>
__global__ void
ExecuteObject(IntegT* integrand, double zt, double lnM, double* result)
{
  *result = integrand->operator()(zt, lnM);
}

TEST_CASE("HMF_t CONDITIONAL MODEL EXECUTION")
{
  double const zt = 0x1.cccccccccccccp-2;
  double const lnM = 0x1.0cp+5;

  typedef quad::hmf_t<GPU> hmftGPU;
  // objection instantiation on CPU
  hmftGPU hmf2 = make_from_file<hmftGPU>("data/HMF_t.dump");

  // need equivalent on GPU memory
  hmftGPU* dhmf2;
  double* result;

  cudaMallocManaged((void**)&dhmf2, sizeof(hmftGPU));
  cudaMallocManaged((void**)&result, sizeof(double));
  *result = 0.;

  // undefined behavior, how to get around this?
  memcpy(dhmf2, &hmf2, sizeof(hmftGPU));
  ExecuteObject<hmftGPU><<<1, 1>>>(dhmf2, lnM, zt, result);
  cudaDeviceSynchronize();
  CHECK(*result != 0.);

  cudaFree(dhmf2);
  cudaFree(result);
}

