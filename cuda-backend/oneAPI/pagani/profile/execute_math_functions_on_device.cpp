// #include <oneapi/dpl/execution>
// #include <oneapi/dpl/async>
#include <CL/sycl.hpp>
// #include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include <string>
// #include <math.h> //initial results were collected with math.h library
#include <cmath> 

class trivial_powr {
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
    return sycl::powr(x, 4.) + sycl::powr(y, 4.) + sycl::powr(z, 4.) +
           sycl::powr(w, 4.) + sycl::powr(v, 4.) + sycl::powr(u, 4.) +
           sycl::powr(t, 4.) + sycl::powr(s, 4.);
  }
};

class trivial_pow {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y)
  {
    return sycl::pow(x, 2.) + sycl::pow(y, 2.);
  }
};

class trivial_pown {
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
    return sycl::pown(x, 4) + sycl::pown(y, 4) + sycl::pown(z, 4) +
           sycl::pown(w, 4) + sycl::pown(v, 4) + sycl::pown(u, 4) +
           sycl::pown(t, 4) + sycl::pown(s, 4);
  }
};

class sycl_pow_double_4 {
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
    return sycl::pow(x, 4.) + sycl::pow(y, 4.) + sycl::pow(z, 4.) +
           sycl::pow(w, 4.) + sycl::pow(v, 4.) + sycl::pow(u, 4.) +
           sycl::pow(t, 4.) + sycl::pow(s, 4.);
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

class sycl_cos_2D{
public:
  SYCL_EXTERNAL double
    operator()(double u, double v)
  {
	return sycl::cos(u) - sycl::cos(v);    
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

class warmpup{
public:
  SYCL_EXTERNAL double
    operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	return u + v + w + x + y + z + p + t;    
  }	
};

int
main(int argc, char** argv)
{
  size_t num_invocs = argc > 1 ? std::stoi(argv[1]) : 100000;
  std::cout << "num_invocs:" << num_invocs << std::endl;
  std::array<double, 8> point = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::array<double, 2> point_2D = {0.1, 0.2};

  double sum = 0.;

  sum += execute_integrand_at_points<warmpup, 8>(num_invocs);
  //sum += execute_integrand_at_points<sycl_pow_double_4, 8>(num_invocs);
  //sum += execute_integrand_at_points<sycl_pow_double_3, 8>(num_invocs);
  //sum += execute_integrand_at_points<sycl_pow_double_2, 8>(num_invocs);
  //sum += execute_integrand_at_points<sycl_pow_int_3, 8>(num_invocs);
  //sum += execute_integrand_at_points<sycl_exp_d, 8>(num_invocs);
  //sum += execute_integrand_at_points<trivial_pow, 2>(num_invocs);
  //sum += execute_integrand_at_points<trivial_powr, 8>(num_invocs);
  //sum += execute_integrand_at_points<trivial_pown, 8>(num_invocs);
  sum += execute_integrand_at_points<sycl_cos, 8>(num_invocs);
  sum += execute_integrand_at_points<sycl_sin, 8>(num_invocs);
  //sum += execute_integrand_at_points<sycl_cos_2D, 2>(num_invocs);


  printf("%.15e\n", sum);

  return 0;
}
