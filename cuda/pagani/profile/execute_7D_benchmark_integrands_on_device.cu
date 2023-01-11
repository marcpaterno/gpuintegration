#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include <array>
#include <cuda_profiler_api.h>
#include "cuda/integrands.cuh"

template<typename F, int ndim>
__global__ void
kernel(F* integrand, double* d_point, double* output, size_t num_invocations){
	size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	double total = 0.;
	gpu::cudaArray<double, ndim> point;
	
	double start_val = .1;
	#pragma unroll 1
	for(int i=0; i < ndim; ++i){
		point[i] = start_val * (i + 1); 
		//point[i] = d_point[i];
	}
	
	#pragma unroll 1
	for(int i=0; i < num_invocations; ++i){
		
		double res = gpu::apply(*integrand, point);
		//double res = integrand->operator()(point[0], point[1], point[2], point[3], point[4], point[5], point[6], point[7]);
		//		double res = point[0] / point[1] / point[2] / point[3] / point[4] / point[5] / point[6] / point[7];

		total += res;
	}
	output[tid] = total;
}


int main(int argc, char** argv){
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  constexpr int ndim = 7;
  double sum = 0.;
  sum += execute_integrand_at_points<F_1_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_2_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_3_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_4_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_5_7D, ndim>(num_invocations);
  sum += execute_integrand_at_points<F_6_7D, ndim>(num_invocations);
  printf("%.15e\n", sum);
  
  return 0;
}
