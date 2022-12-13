#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include <array>
#include <cuda_profiler_api.h>
#include "cuda/integrands.cuh"

/*
class F_2_7D {
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t)
  {
	//return x / y / z / w / v / u / t / s;
  
	const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / pow(a, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(a, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(a, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(a, 2.)) + pow(w - b, 2.));
    const double term_5 = 1. / ((1. / pow(a, 2.)) + pow(v - b, 2.));
    const double term_6 = 1. / ((1. / pow(a, 2.)) + pow(u - b, 2.));
	const double term_7 = 1. / ((1. / pow(a, 2.)) + pow(t - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7;
    return val;
  }
};

class F_3_7D{
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t)
  {
	//return x / y / z / w / v / u / t / s;
  
	return pow(1. + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_7D {
  public:
    __device__ __host__ double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t)
    {
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.) + 
				pow(25., 2.) * pow(u - beta, 2.) + 
				pow(25., 2.) * pow(t - beta, 2.)));
    }
};

class F_5_7D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p)
  {
	//return x / y / z / k / m / n / p / q;
  
	double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta);
    return exp(t1);
  }
};

class F_6_7D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z, double p)
  {
	//return u / v / w / x / y / z / p / t;
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. *p);
  }
};
*/

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
  std::array<double, ndim> point = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7};
  double sum = 0.;
  sum += execute_integrand<F_2_7D, ndim>(point, num_invocations);
  sum += execute_integrand<F_3_7D, ndim>(point, num_invocations);
  sum += execute_integrand<F_4_7D, ndim>(point, num_invocations);
  sum += execute_integrand<F_5_7D, ndim>(point, num_invocations);
  sum += execute_integrand<F_6_7D, ndim>(point, num_invocations);
  printf("%.15e\n", sum);
  
  return 0;
}
