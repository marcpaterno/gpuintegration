#include <stdio.h>

__global__ void
cuda_hello()
{
  printf("Hello World from GPU!\n");
}

int
main()
{
  printf("Before calling the kernel\n");
  cuda_hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("After calling the kernel\n");
  return 0;
}
