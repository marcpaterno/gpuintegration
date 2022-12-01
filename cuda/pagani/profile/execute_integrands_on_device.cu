#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include <array>

class GENZ_2_8D {
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
	const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / pow(a, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(a, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(a, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(a, 2.)) + pow(w - b, 2.));
    const double term_5 = 1. / ((1. / pow(a, 2.)) + pow(v - b, 2.));
    const double term_6 = 1. / ((1. / pow(a, 2.)) + pow(u - b, 2.));
	const double term_7 = 1. / ((1. / pow(a, 2.)) + pow(t - b, 2.));
	const double term_8 = 1. / ((1. / pow(a, 2.)) + pow(s - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7 * term_8;
    return val;
  }
};

class GENZ_3_8D{
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
	return pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class GENZ_4_8D {
  public:
    __device__ __host__ double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
    {
	  double beta = .5;
      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.) + 
				pow(25., 2.) * pow(u - beta, 2.) + 
				pow(25., 2.) * pow(t - beta, 2.) + 
				pow(25., 2.) * pow(s - beta, 2.)));
    }
};

class GENZ_5_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
	double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta) - 10. * fabs(q - beta);
    return exp(t1);
  }
};

class GENZ_6_8D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3 || t > .2)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. *p + 3. * t);
  }
};

template<typename F>
__global__ void
kernel(F* integrand, double* d_point, double* output, size_t num_invocations){
	size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	double total = 0.;
	gpu::cudaArray<double, 8> point;
	
	for(int i=0; i < 8; ++i){
		point[i] = d_point[i];
	}
	
	for(int i=0; i < num_invocations; ++i){
		
		double res = gpu::apply(*integrand, point);
		//double res = integrand->operator()(point[0], point[1], point[2], point[3], point[4], point[5], point[6], point[7]);
		//		double res = point[0] / point[1] / point[2] / point[3] / point[4] / point[5] / point[6] / point[7];

		total += res;
	}
	output[tid] = total;
}

template<typename F>
double execute_integrand(std::array<double, 8> point){
	const size_t num_blocks = 1024;
	const size_t num_threads = 64;
	const size_t num_invocations = 10000;
	
	F integrand;  
	F* d_integrand = quad::cuda_copy_to_managed<F>(integrand);
	
	double* d_point = quad::cuda_malloc<double>(point.size());
	quad::cuda_memcpy_to_device(d_point, point.data(), point.size());
		
	double* output = quad::cuda_malloc<double>(num_threads*num_blocks);
	
	for(int i = 0; i < 10; ++i)
	{
		kernel<F><<<num_blocks, num_threads>>>(d_integrand, d_point, output, num_invocations);
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
}

int
main()
{
  std::array<double, 8> point = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8};
  double sum = 0.;

  sum += execute_integrand<GENZ_2_8D>(point);
  sum += execute_integrand<GENZ_3_8D>(point);
  sum += execute_integrand<GENZ_4_8D>(point);
  sum += execute_integrand<GENZ_5_8D>(point);
  sum += execute_integrand<GENZ_6_8D>(point);
  printf("%.15e\n", sum);
  
  return 0;
}
