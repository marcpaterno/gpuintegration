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

TEST_CASE("Cubes distributed evenly to threads")
{	
	const double ncall = 1.e6;
	const int ndim = 6;
    std::array<int, 3> chunkSizes = {2048, 1024, 32};
	
    for (auto chunkSize : chunkSizes) 
	  {
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
		
		double computed_sum = 0.;
		
		for (uint32_t i = 0; i < kernel_params.nBlocks; ++i) {
		
			bool last_block = i == (kernel_params.nBlocks - 1);
			bool last_block_aligned = kernel_params.totalNumThreads % 128 == 0;
			double true_val = last_block & !last_block_aligned ? (kernel_params.totalNumThreads % 128) * fgb : BLOCK_DIM_X * fgb;	 
			computed_sum += block_reductions[i];
			CHECK(block_reductions[i] == Approx(true_val));
		}

		
		CHECK(computed_sum == Approx(kernel_params.totalNumThreads * fgb));
        CHECK(result == Approx(kernel_params.totalNumThreads * fgb));
		
      }
}


TEST_CASE("Cubes uneven among threads")
{
  const int ndim = 7;
  double ncall = 1.e9;
  
  std::array<int, 3> chunkSizes = {2048, 1024, 32};

      for (auto chunkSize : chunkSizes) 
	  {
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
		
		double computed_sum = 0.;
		
		for (uint32_t i = 0; i < kernel_params.nBlocks; ++i) {
		
			bool last_block = i == (kernel_params.nBlocks - 1);
			bool last_block_aligned = kernel_params.totalNumThreads % 128 == 0;
			double true_val = last_block & !last_block_aligned ? (kernel_params.totalNumThreads % 128) * fgb : BLOCK_DIM_X * fgb;	 
			computed_sum += block_reductions[i];
			CHECK(block_reductions[i] == Approx(true_val));
		}

		
		CHECK(computed_sum == Approx(kernel_params.totalNumThreads * fgb));
        CHECK(result == Approx(kernel_params.totalNumThreads * fgb));
		
      }
    
  
}