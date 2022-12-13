// #include <oneapi/dpl/execution>
// #include <oneapi/dpl/async>
#include <CL/sycl.hpp>
// #include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include <string>
// #include <math.h>
#include <cmath>



class trivial_pow {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y)
  {
    return sycl::pow(x, 2.) + y;
  }
};

class trivial_pown {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y)
  {
    return sycl::pown(x, 2) + y;
  }
};

class sycl_pow_double_3 {
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
    return sycl::pow(x, 3.) + sycl::pow(y, 3.) + sycl::pow(z, 3.) +
           sycl::pow(w, 3.) + sycl::pow(v, 3.) + sycl::pow(u, 3.) +
           sycl::pow(t, 3.) + sycl::pow(s, 3.);
  }
};

class sycl_pow_double_2 {
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
    return sycl::pow(x, 2.) + sycl::pow(y, 2.) + sycl::pow(z, 2.) +
           sycl::pow(w, 2.) + sycl::pow(v, 2.) + sycl::pow(u, 2.) +
           sycl::pow(t, 2.) + sycl::pow(s, 2.);
  }
};

class sycl_pow_int_3 {
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
    return sycl::pown(x, 3) - sycl::pown(y, 3) - sycl::pown(z, 3) -
           sycl::pown(w, 3) - sycl::pown(v, 3) - sycl::pown(u, 3) -
           sycl::pown(t, 3) - sycl::pown(s, 3);
  }
};

class std_pow_double_2 {
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
    return std::pow(x, 2.) + std::pow(y, 2.) + std::pow(z, 2.) +
           std::pow(w, 2.) + std::pow(v, 2.) + std::pow(u, 2.) +
           std::pow(t, 2.) + std::pow(s, 2.);
  }
};

class clang_powf_double_2 {
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
    return powf(x, 2.) + powf(y, 2.) + powf(z, 2.) + powf(w, 2.) + powf(v, 2.) +
           powf(u, 2.) + powf(t, 2.) + powf(s, 2.);
  }
};

class sycl_exp_d {
public:
  SYCL_EXTERNAL double
  operator()(double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double p,
             double t)
  {
    return sycl::exp(u) + sycl::exp(v) + sycl::exp(w) + sycl::exp(x) +
           sycl::exp(y) + sycl::exp(z) + sycl::exp(p) + sycl::exp(t);
  }
};

class sycl_cos{
public:
  SYCL_EXTERNAL double
    operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	return sycl::cos(u) - sycl::cos(v) - sycl::cos(w) - sycl::cos(x) - sycl::cos(y) - sycl::cos(z) - sycl::cos(p) - sycl::cos(t);    
  }
};

class sycl_sin{
public:
  SYCL_EXTERNAL double
    operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	return sycl::sin(u) - sycl::sin(v) - sycl::sin(w) - sycl::sin(x) - sycl::sin(y) - sycl::sin(z) - sycl::sin(p) - sycl::sin(t);    
  }
};


int
main(int argc, char** argv)
{
  size_t num_invocations = argc > 1 ? std::stoi(argv[1]) : 100000;
  std::cout << "num_invocations:" << num_invocations << std::endl;
  std::array<double, 8> point = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::array<double, 2> point_2D = {0.1, 0.2};

  double sum = 0.;

  // this that yield a backend error
  // the std equivalents only work with cmath, not math.h
  /*
  ///this  means that there is a specialization for powf(x,2)
   sum += execute_integrand<Powf_to_third_add_double>(point, num_invocations);
   sum += execute_integrand<Powf_to_third_add_int>(point, num_invocations);
   sum += execute_integrand<Expf_sub>(point, num_invocations);
   sum += execute_integrand<Expf_add>(point, num_invocations);
   sum += execute_integrand<Expf_div>(point, num_invocations);
   sum += execute_integrand<Std_Pow_to_third_add_double>(point,
  num_invocations); sum += execute_integrand<Std_Pow_to_third_add_int>(point,
  num_invocations);

   sum += execute_integrand<Std_Exp_sub>(point, num_invocations);
   sum += execute_integrand<Std_Exp_add>(point, num_invocations);
   sum += execute_integrand<Std_Exp_div>(point, num_invocations);
  */

  sum +=
    execute_integrand<sycl_pow_double_3, 8>(point, num_invocations);
  sum +=
    execute_integrand<sycl_pow_double_2, 8>(point, num_invocations);
  sum +=
    execute_integrand<sycl_pow_int_3, 8>(point, num_invocations);

  //sum += execute_integrand<std_pow_double_2>(point, num_invocations);
  //sum += execute_integrand<clang_powf_double_2>(point, num_invocations);

  sum += execute_integrand<sycl_exp_d, 8>(point, num_invocations);
  sum += execute_integrand<trivial_pow, 2>(point_2D, num_invocations);
  sum += execute_integrand<trivial_pown, 2>(point_2D, num_invocations);
  sum += execute_integrand<sycl_cos, 8>(point, num_invocations);
  sum += execute_integrand<sycl_sin, 8>(point, num_invocations);


  printf("%.15e\n", sum);

  return 0;
}
