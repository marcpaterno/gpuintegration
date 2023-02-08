#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"
#include <array>

class trivial_powf {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    return powf(x, 2.) + powf(y, 2.);
  }
};

class trivial_pow {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    return pow(x, 2.) + pow(y, 2.);
  }
};

class nvcc_powf_int_3 {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return powf(x, 3) - powf(y, 3) - powf(z, 3) - powf(w, 3) - powf(v, 3) -
           powf(u, 3) - powf(t, 3) - powf(s, 3);
  }
};

class nvcc_powf_double_3 {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return powf(x, 3.) + powf(y, 3.) + powf(z, 3.) + powf(w, 3.) + powf(v, 3.) +
           powf(u, 3.) + powf(t, 3.) + powf(s, 3.);
  }
};

class nvcc_powf_double_2 {
public:
  KOKKOS_INLINE_FUNCTION double
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

class nvcc_powf_int_2 {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return powf(x, 2) + powf(y, 2) + powf(z, 2) + powf(w, 2) + powf(v, 2) +
           powf(u, 2) + powf(t, 2) + powf(s, 2);
  }
};

class nvcc_pow_double_4 {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return pow(x, 4.) + pow(y, 4.) + pow(z, 4.) + pow(w, 4.) + pow(v, 4.) +
           pow(u, 4.) + pow(t, 4.) + pow(s, 4.);
  }
};

class nvcc_pow_double_3 {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return pow(x, 3.) + pow(y, 3.) + pow(z, 3.) + pow(w, 3.) + pow(v, 3.) +
           pow(u, 3.) + pow(t, 3.) + pow(s, 3.);
  }
};

class nvcc_pow_double_2 {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return pow(x, 2.) + pow(y, 2.) + pow(z, 2.) + pow(w, 2.) + pow(v, 2.) +
           pow(u, 2.) + pow(t, 2.) + pow(s, 2.);
  }
};

class nvcc_pow_int_3 {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return pow(x, 3) - pow(y, 3) - pow(z, 3) - pow(w, 3) - pow(v, 3) -
           pow(u, 3) - pow(t, 3) - pow(s, 3);
  }
};

class nvcc_pow_int_2 {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return pow(x, 2) - pow(y, 2) - pow(z, 2) - pow(w, 2) - pow(v, 2) -
           pow(u, 2) - pow(t, 2) - pow(s, 2);
  }
};

class nvcc_exp_d {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double p,
             double t)
  {
    return exp(u) + exp(v) + exp(w) + exp(x) + exp(y) + exp(z) + exp(p) +
           exp(t);
  }
};

class nvcc_expf_d {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double p,
             double t)
  {
    return expf(u) - expf(v) - expf(w) - expf(x) - expf(y) - expf(z) - expf(p) -
           expf(t);
  }
};

class nvcc_cos {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double p,
             double t)
  {
    return cos(u) - cos(v) - cos(w) - cos(x) - cos(y) - cos(z) - cos(p) -
           cos(t);
  }
};

class nvcc_cos_2D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double u, double v)
  {
    return cos(u) - cos(v);
  }
};

class nvcc_sin {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double p,
             double t)
  {
    return sin(u) - sin(v) - sin(w) - sin(x) - sin(y) - sin(z) - sin(p) -
           sin(t);
  }
};

class warmpup {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double p,
             double t)
  {
    return u + v + w + x + y + z + p + t;
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  size_t num_invocs = argc > 1 ? std::stoi(argv[1]) : 100000;
  double sum = 0.;

  sum += execute_integrand_at_points<nvcc_powf_int_3, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_powf_double_3, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_powf_double_2, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_powf_int_2, 8>(num_invocs);
  sum += execute_integrand_at_points<warmpup, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_pow_double_4, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_pow_double_3, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_pow_double_2, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_pow_int_3, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_pow_int_2, 8>(num_invocs);

  sum += execute_integrand_at_points<nvcc_exp_d, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_expf_d, 8>(num_invocs);

  sum += execute_integrand_at_points<trivial_pow, 2>(num_invocs);
  sum += execute_integrand_at_points<trivial_powf, 2>(num_invocs);

  sum += execute_integrand_at_points<nvcc_cos, 8>(num_invocs);
  sum += execute_integrand_at_points<nvcc_cos_2D, 2>(num_invocs);
  sum += execute_integrand_at_points<nvcc_sin, 8>(num_invocs);

  printf("%.15e\n", sum);
  Kokkos::finalize();
  return 0;
}
