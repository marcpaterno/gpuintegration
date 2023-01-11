#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include <array>
#include <cuda_profiler_api.h>
#include "cuda/integrands.cuh"

/*
class F_2_6D {
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
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

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

class F_3_6D{
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
  {
	//return x / y / z / w / v / u / t / s;
  
	return pow(1. + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_6D {
  public:
    __device__ __host__ double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
    {
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.) + 
				pow(25., 2.) * pow(u - beta, 2.)));
    }
};

class F_5_6D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n)
  {
	//return x / y / z / k / m / n / p / q;
  
	double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta);
    return exp(t1);
  }
};

class F_6_6D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z)
  {
	//return u / v / w / x / y / z / p / t;
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }
};
*/


int main(int argc, char** argv){
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  constexpr int ndim = 6;
  std::array<double, ndim> point = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6};
  double sum = 0.;
  sum += execute_integrand<F_2_6D, ndim>(point, num_invocations);
  sum += execute_integrand<F_3_6D, ndim>(point, num_invocations);
  sum += execute_integrand<F_4_6D, ndim>(point, num_invocations);
  sum += execute_integrand<F_5_6D, ndim>(point, num_invocations);
  sum += execute_integrand<F_6_6D, ndim>(point, num_invocations);
  printf("%.15e\n", sum);
  
  return 0;
}