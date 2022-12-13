//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"



class F_2_6D {
public:
  SYCL_EXTERNAL double
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
    const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
    const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
    const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
    const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(w - b, 2.));
    const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(v - b, 2.));
    const double term_6 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(u - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

class F_3_6D{
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
  {
	//return x / y / z / w / v / u / t / s;
	return sycl::pow(1. + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_6D {
  public:
    SYCL_EXTERNAL double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
    {
	  //return x / y / z / w / v / u / t / s;

	  double beta = .5;
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(u - beta, 2.)));
    }
};

class F_5_6D {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n)
  {
	//return x / y / z / k / m / n / p / q;
	double beta = .5;
    double t1 = -10. * sycl::fabs(x - beta) - 10. * sycl::fabs(y - beta) -
                10. * sycl::fabs(z - beta) - 10. * sycl::fabs(k - beta) -
                10. * sycl::fabs(m - beta) - 10. * sycl::fabs(n - beta);
    return sycl::exp(t1);
  }
};

class F_6_6D {
public:
  SYCL_EXTERNAL double
  operator()(double u, double v, double w, double x, double y, double z)
  {
	//return u / v / w / x / y / z / p / t;
  
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return sycl::exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }
};

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

