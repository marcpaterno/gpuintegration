//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"

class F_2_8D {
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
	//return x / y / z / w / v / u / t / s;
  
	const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
    const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
    const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
    const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(w - b, 2.));
    const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(v - b, 2.));
    const double term_6 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(u - b, 2.));
	const double term_7 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(t - b, 2.));
	const double term_8 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(s - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7 * term_8;
    return val;
  }
};

class F_3_8D{
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
	//return x / y / z / w / v / u / t / s;
	return sycl::pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_8D {
  public:
    SYCL_EXTERNAL double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
    {
	  //return x / y / z / w / v / u / t / s;

	  double beta = .5;
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(u - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(t - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(s - beta, 2.)));
    }
};

class F_5_8D {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
	//return x / y / z / k / m / n / p / q;
	double beta = .5;
    double t1 = -10. * sycl::fabs(x - beta) - 10. * sycl::fabs(y - beta) -
                10. * sycl::fabs(z - beta) - 10. * sycl::fabs(k - beta) -
                10. * sycl::fabs(m - beta) - 10. * sycl::fabs(n - beta) -
                10. * sycl::fabs(p - beta) - 10. * sycl::fabs(q - beta);
    return sycl::exp(t1);
  }
};

class F_6_8D {
public:
  SYCL_EXTERNAL double
  operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	//return u / v / w / x / y / z / p / t;
  
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3 || t > .2)
      return 0.;
    else
      return sycl::exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. *p + 3. * t);
  }
};


int main(int argc, char** argv){
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  std::array<double, 8> point = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8};
  double sum = 0.;
  sum += execute_integrand<F_2_8D, 8>(point, num_invocations);
  sum += execute_integrand<F_3_8D, 8>(point, num_invocations);
  sum += execute_integrand<F_4_8D, 8>(point, num_invocations);
  sum += execute_integrand<F_5_8D, 8>(point, num_invocations);
  sum += execute_integrand<F_6_8D, 8>(point, num_invocations);
  printf("%.15e\n", sum);
  
    return 0;
}

