#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

__global__ void
profile(double* block_results){
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  double val = static_cast<double>(tid);
  
  if(blockIdx.x == 0 && threadIdx.x == 0)
	printf("block %i : %e\n", blockIdx.x, val);

  __syncthreads();
  val = quad::blockReduceSum(val);
  __syncthreads();
  
  if(threadIdx.x == 0)
	block_results[blockIdx.x] = val;

}

double compute_expected(size_t num_blocks, size_t num_threads){
	size_t res = 0;
	for(int i=0; i < num_blocks * num_threads; ++i)
		res += i;
	return static_cast<double>(res);
}

int
main()
{
  size_t num_blocks = 262144*4;
  size_t num_threads = 64;
  double* block_res = cuda_malloc<double>(num_blocks);
  profile<<<num_blocks, num_threads>>>(block_res);
  cudaDeviceSynchronize();
  double res = reduction<double>(block_res, num_blocks);
  printf("res:%e expected:%e\n", res, compute_expected(num_blocks, num_threads));
  return 0;
}
