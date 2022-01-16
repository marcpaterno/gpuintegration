#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"
#include "vegas/util/util.cuh"
#include "vegas/util/vegas_utils.cuh"
#include "vegas/vegasT.cuh"
#include <array>

__global__ void
kernel(int chunkSize,
       uint32_t totalNumThreads,
       int LastChunk,
       double* result_dev,
       double* block_reductions)
{
  uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t tx = threadIdx.x;
  double fbg = 0.; // each block reduction yields 128*1.1 even though the second
                   // block shouldn'tbe getting it from all
  if (m < totalNumThreads) {
    fbg = 1.1;
    if (m == totalNumThreads - 1)
      chunkSize = LastChunk + 1;
  }

  fbg = cuda_mcubes::blockReduceSum(fbg); // i thought issue would be ere

  if (tx == 0) {
    block_reductions[blockIdx.x] = fbg;
    atomicAdd(&result_dev[0],
              fbg); // i thought this wouldn't be problematic, threads where m >
                    // totalNumThreads should have zeroes all over
  }
}

void
Reduction(double ncall, int chunkSize, int ndim)
{
  Kernel_Params kernel_params(ncall, chunkSize, ndim);
  double* result = cuda_malloc_managed<double>(1);
  double* block_reductions =
    cuda_malloc_managed<double>(static_cast<size_t>(kernel_params.nBlocks));
  double fgb = 1.1;

  kernel<<<kernel_params.nBlocks, kernel_params.nThreads>>>(
    chunkSize,
    kernel_params.totalNumThreads,
    kernel_params.LastChunk,
    result,
    block_reductions);
  cudaDeviceSynchronize();

  SECTION("Block Reduction")
  {
    for (uint32_t i = 0; i < kernel_params.nBlocks - 1; i++) {
      CHECK(block_reductions[i] == Approx(BLOCK_DIM_X * fgb));
    }
  }

  SECTION("Atomic Addition")
  {
    CHECK(result[0] == Approx(kernel_params.totalNumThreads * fgb));
  }

  // check last thread which doesn't have the same chunks
}

TEST_CASE("Reduction")
{
  std::array<double, 6> ncalls = {1.e6, 1.e7, 1.e8, 1.e9, 2.e9, 4.e9};
  std::array<int, 7> ndims = {2, 3, 4, 5, 6, 7, 8};
  std::array<int, 4> chunkSizes = {2048, 1024, 32, 2};

  for (auto ncall : ncalls)
    for (auto ndim : ndims)
      for (auto chunkSize : chunkSizes) {
        Reduction(ncall, chunkSize, ndim);
      }
}
