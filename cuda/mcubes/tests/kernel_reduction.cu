#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"
#include "cuda/mcubes/util/util.cuh"
#include "cuda/mcubes/util/vegas_utils.cuh"
#include "cuda/mcubes/vegasT.cuh"
#include "cuda/pagani/quad/util/mem_util.cuh"
#include <array>

__global__ void
run_block_reduce_sum(int32_t totalNumThreads,
                     double* total_sum,
                     double* block_reductions)
{
  auto const block_id = blockIdx.x;
  auto const global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  auto const block_thread_id = threadIdx.x;

  double const fbg = (global_thread_id < totalNumThreads) ? 1.1 : 0.0;
  double const sum = cuda_mcubes::blockReduceSum(fbg);

  if (block_thread_id == 0) {
    // Write this block's answer to global memory.
    block_reductions[block_id] = sum;
    // i thought this wouldn't be problematic, threads where m > totalNumThreads
    // should have zeroes all over
    // Every block contributes to result_dev
    atomicAdd(total_sum, sum);
  }
}

TEST_CASE("Aligned reduction")
{
  std::array<double, 6> ncalls = {1.e6, 1.e7, 1.e8, 1.e9, 2.e9, 4.e9};
  std::array<int, 7> ndims = {2, 3, 4, 5, 6, 7, 8};
  std::array<int, 3> chunkSizes = {2048, 1024, 32};

  for (auto ncall : ncalls) {
    for (auto ndim : ndims) {
      for (auto chunkSize : chunkSizes) {
        Kernel_Params kernel_params(ncall, chunkSize, ndim);

        double* d_result = cuda_malloc<double>(1);

        double* block_reductions = cuda_malloc_managed<double>(
          static_cast<size_t>(kernel_params.nBlocks));
        double fgb = 1.1;

        run_block_reduce_sum<<<kernel_params.nBlocks, kernel_params.nThreads>>>(
          kernel_params.totalNumThreads,
          d_result,
          block_reductions);
        cudaDeviceSynchronize();

        double result = 0.0;
        cuda_memcpy_to_host(&result, d_result, 1); // newly allocated

        for (uint32_t i = 0; i < kernel_params.nBlocks - 1; i++) {
          CHECK(block_reductions[i] == Approx(BLOCK_DIM_X * fgb));
        }

        CHECK(result == Approx(kernel_params.totalNumThreads * fgb));
      }
    }
  }
}

TEST_CASE("misaigned reduction")
{
  // do something here
}
