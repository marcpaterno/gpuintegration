#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include <array>

		class trivial_powf{
			public:
			__host__ __device__ double
			operator()(double x, double y){
				return powf(x, 2.) + y;
			}
		};
		
		class trivial_pow{
			public:
			__host__ __device__ double
			operator()(double x, double y){
				return pow(x, 2.) + y;
			}
		};
		
		class nvcc_powf_int_3{
			public:
			__host__ __device__ double
			operator()(double x,
						double y,
						double z,
						double w,
						double v,
						double u,
						double t,
						double s)
			{
				return powf(x, 3) - powf(y, 3) - powf(z, 3) - powf(w, 3) - powf(v, 3) - powf(u, 3) - powf(t, 3) - powf(s, 3);
			}
		};
				
		class nvcc_powf_double_3{
		public:
		  __host__ __device__ double
		  operator()(double x,
					   double y,
					   double z,
					   double w,
					   double v,
					   double u,
					   double t,
					   double s)
		  {
			return powf(x, 3.) + powf(y, 3.) + powf(z, 3.) + powf(w, 3.) + powf(v, 3.) + powf(u, 3.) + powf(t, 3.) + powf(s, 3.);    
		  }
		};
		
		class nvcc_powf_double_2 {
			public:
			__host__ __device__ double
			operator()(double x,
				   double y,
				   double z,
				   double w,
				   double v,
				   double u,
				   double t,
				   double s)
			{
			return powf(x, 2.) + powf(y, 2.) + powf(z, 2.) + powf(w, 2.) + powf(v, 2.) + powf(u, 2.) + powf(t, 2.) + powf(s, 2.);
			}
		};

		class nvcc_powf_int_2 {
			public:
			__host__ __device__ double
			operator()(double x,
				   double y,
				   double z,
				   double w,
				   double v,
				   double u,
				   double t,
				   double s)
			{
			return powf(x, 2) + powf(y, 2) + powf(z, 2) + powf(w, 2) + powf(v, 2) + powf(u, 2) + powf(t, 2) + powf(s, 2);
			}
		};

		class nvcc_pow_double_3{
			public:
		  __host__ __device__ double
		  operator()(double x,
					   double y,
					   double z,
					   double w,
					   double v,
					   double u,
					   double t,
					   double s)
		  {
			return pow(x, 3.) + pow(y, 3.) + pow(z, 3.) + pow(w, 3.) + pow(v, 3.) + pow(u, 3.) + pow(t, 3.) + pow(s, 3.);    
		  }
		};	
		
		class nvcc_pow_double_2 {
		  public:
			__host__ __device__ double
			operator()(double x,
					   double y,
					   double z,
					   double w,
					   double v,
					   double u,
					   double t,
					   double s)
			{
			  return pow(x, 2.) + pow(y, 2.) + pow(z, 2.) + pow(w, 2.) + pow(v, 2.) + pow(u, 2.) + pow(t, 2.) + pow(s, 2.);
			}
		};

		class nvcc_pow_int_3{
			public:
			__host__ __device__ double
			operator()(double x,
					   double y,
					   double z,
					   double w,
					   double v,
					   double u,
					   double t,
					   double s)
			  {
				return pow(x, 3) - pow(y, 3) - pow(z, 3) - pow(w, 3) - pow(v, 3) - pow(u, 3) - pow(t, 3) - pow(s, 3);
			  }
			};

		class nvcc_pow_int_2{
			public:
			__host__ __device__ double
			operator()(double x,
					   double y,
					   double z,
					   double w,
					   double v,
					   double u,
					   double t,
					   double s)
			  {
				return pow(x, 2) - pow(y, 2) - pow(z, 2) - pow(w, 2) - pow(v, 2) - pow(u, 2) - pow(t, 2) - pow(s, 2);
			  }
			};

class nvcc_exp_d{
public:
  __host__ __device__ double
  operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	return exp(u) + exp(v) + exp(w) + exp(x) + exp(y) + exp(z) + exp(p) + exp(t);
  }
};

class nvcc_expf_d{
public:
  __host__ __device__ double
    operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	return expf(u) - expf(v) - expf(w) - expf(x) - expf(y) - expf(z) - expf(p) - expf(t);    
  }
};

class nvcc_cos{
public:
  __host__ __device__ double
    operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	return cos(u) - cos(v) - cos(w) - cos(x) - cos(y) - cos(z) - cos(p) - cos(t);    
  }
};

class nvcc_sin{
public:
  __host__ __device__ double
    operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	return sin(u) - sin(v) - sin(w) - sin(x) - sin(y) - sin(z) - sin(p) - sin(t);    
  }
};




/*template<typename F, int ndim>
__global__ void
kernel(F* integrand, double* d_point, double* output, size_t num_invocations){
	size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	double total = 0.;
	gpu::cudaArray<double, ndim> point;
	
	#pragma unroll 1
	for(int i=0; i < ndim; ++i){
		point[i] = d_point[i];
	}
	
	#pragma unroll 1
	for(int i=0; i < num_invocations; ++i){
		double res = gpu::apply(*integrand, point);
		total += res;
	}
	output[tid] = total;
}

template<typename F, int ndim>
double execute_integrand(std::array<double, ndim> point, size_t num_invocations){
	const size_t num_blocks = 1024;
	const size_t num_threads = 64;
	//const size_t num_invocations = 100000;
	
	F integrand;  
	F* d_integrand = quad::cuda_copy_to_managed<F>(integrand);
	
	double* d_point = quad::cuda_malloc<double>(point.size());
	quad::cuda_memcpy_to_device(d_point, point.data(), point.size());
		
	double* output = quad::cuda_malloc<double>(num_threads*num_blocks);
	
	
	for(int i = 0; i < 10; ++i)
	{
		kernel<F, ndim><<<num_blocks, num_threads>>>(d_integrand, d_point, output, num_invocations);
		cudaDeviceSynchronize();
	}
	
	std::vector<double> host_output;
	host_output.resize(num_threads*num_blocks);
	//std::cout<<"vector size:"<<host_output.size()<<std::endl;
	cuda_memcpy_to_host<double>(host_output.data(), output, host_output.size());
	
	double sum = 0.;
	for(int i=0; i < num_threads*num_blocks; ++i)
		sum += host_output[i];
	
	cudaFree(output);
	cudaFree(d_integrand);
	cudaFree(d_point);
	return sum;
}*/

int main(int argc, char** argv){
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  std::array<double, 8> point = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8};
  std::array<double, 2> point_2D = {0.1, 0.2};

  double sum = 0.;

  sum += execute_integrand<nvcc_powf_int_3, 8>(point, num_invocations);
  sum += execute_integrand<nvcc_powf_double_3, 8>(point, num_invocations);
  sum += execute_integrand<nvcc_powf_double_2, 8>(point, num_invocations);
  sum += execute_integrand<nvcc_powf_int_2, 8>(point, num_invocations);
  

  sum += execute_integrand<nvcc_pow_double_3, 8>(point, num_invocations);
  sum += execute_integrand<nvcc_pow_double_2, 8>(point, num_invocations);
  sum += execute_integrand<nvcc_pow_int_3, 8>(point, num_invocations);
  sum += execute_integrand<nvcc_pow_int_2, 8>(point, num_invocations);
  
  std::cout<<"8 exp"<<std::endl;

  sum += execute_integrand<nvcc_exp_d, 8>(point, num_invocations);
  std::cout<<"8 expf"<<std::endl;
  sum += execute_integrand<nvcc_expf_d, 8>(point, num_invocations);

  std::cout<<"two pows"<<std::endl;
  sum += execute_integrand<trivial_pow, 2>(point_2D, num_invocations);
  std::cout<<"two powfs"<<std::endl;
  sum += execute_integrand<trivial_powf,2>(point_2D, num_invocations);
  sum += execute_integrand<nvcc_cos,8>(point, num_invocations);
  sum += execute_integrand<nvcc_sin,8>(point, num_invocations);


  printf("%.15e\n", sum);
  
  return 0;
}
