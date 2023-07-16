#ifndef KOKKOS_INTREGRANDS_CUH
#define KOKKOS_INTEGRANDS_CUH
#include "cuda/pagani/demos/compute_genz_integrals.cuh"
#include <Kokkos_Core.hpp>

class Addition_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
    return x + y + z + k + m + n + p + q;
  }
};

class Addition_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p)
  {
    return x + y + z + k + m + n + p;
  }
};

class Addition_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v, double m)
  {
    return x + y + z + w + v + m;
  }
};

class Addition_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v)
  {
    return x + y + z + w + v;
  }
};

class Addition_4D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w)
  {
    return x + y + z + w;
  }
};

class Addition_3D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z)
  {
    return x + y + z;
  }
};

class SinSum_3D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z)
  {
    return sin(x + y + z);
  }
};

class SinSum_4D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double k)
  {
    return sin(x + y + z + k);
  }
};

class SinSum_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double k, double l)
  {
    return sin(x + y + z + k + l);
  }
};

class SinSum_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

class SinSum_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n)
  {
    return sin(x + y + z + k + l + m + n);
  }
};

class SinSum_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double p)
  {
    return sin(x + y + z + k + l + m + n + p);
  }
};
class F_1_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
  {
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y +
               8. * z);
  }

  void
  set_true_value()
  {
    true_value = 3.439557952183252e-05;
  }

  double true_value;
};

class F_2_8D {
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

    double val =
      term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7 * term_8;
    return val;
  }

  void
  set_true_value()
  {
    true_value =
      compute_product_peak<8>({50., 50., 50., 50., 50., 50., 50., 50.},
                              {.5, .5, .5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_3_8D {
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
    // correct answer: 2.2751965817917756076e-10
    return pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x +
                 2. * y + z,
               -9.);
  }

  void
  set_true_value()
  {
    true_value = 2.2751965817917756076e-10;
  }

  double true_value;
};

class F_4_8D {
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
    // return x / y / z / w / v / u / t / s;
    double beta = .5;
    return exp(
      -1.0 *
      (pow(25., 2.) * pow(x - beta, 2.) + pow(25., 2.) * pow(y - beta, 2.) +
       pow(25., 2.) * pow(z - beta, 2.) + pow(25., 2.) * pow(w - beta, 2.) +
       pow(25., 2.) * pow(v - beta, 2.) + pow(25., 2.) * pow(u - beta, 2.) +
       pow(25., 2.) * pow(t - beta, 2.) + pow(25., 2.) * pow(s - beta, 2.)));
  }

  void
  set_true_value()
  {
    true_value = compute_gaussian<8>({25., 25., 25., 25., 25., 25., 25., 25.},
                                     {.5, .5, .5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_5_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
    // return x / y / z / k / m / n / p / q;

    double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta) - 10. * fabs(q - beta);
    return exp(t1);
  }

  void
  set_true_value()
  {
    true_value = compute_c_zero<8>({10., 10., 10., 10., 10., 10., 10., 10.},
                                   {.5, .5, .5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_6_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double t,
             double p,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
  {
    if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3 ||
        t > .2)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. * p +
                 3. * t);
  }

  void
  set_true_value()
  {
    true_value = compute_discontinuous<8>({3., 4., 5., 6., 7., 8., 9., 10.},
                                          {.2, .3, .4, .5, .6, .7, .8, .9});
  }

  double true_value;
};

class F_1_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y)
  {
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y);
  }

  void
  set_true_value()
  {
    true_value = -0.00003764562579508779;
  }

  double true_value;
};

class F_2_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t)
  {
    // return x / y / z / w / v / u / t / s;

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

  void
  set_true_value()
  {
    true_value = compute_product_peak<7>({50., 50., 50., 50., 50., 50., 50.},
                                         {.5, .5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_3_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t)
  {
    // correct answer: 1.459429e-08
    return pow(1. + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z,
               -8.);
  }

  void
  set_true_value()
  {
    true_value = 1.459429e-08;
  }

  double true_value;
};

class F_4_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t)
  {
    // return x / y / z / w / v / u / t / s;
    double beta = .5;
    return exp(
      -1.0 *
      (pow(25., 2.) * pow(x - beta, 2.) + pow(25., 2.) * pow(y - beta, 2.) +
       pow(25., 2.) * pow(z - beta, 2.) + pow(25., 2.) * pow(w - beta, 2.) +
       pow(25., 2.) * pow(v - beta, 2.) + pow(25., 2.) * pow(u - beta, 2.) +
       pow(25., 2.) * pow(t - beta, 2.)));
  }

  void
  set_true_value()
  {
    true_value = compute_gaussian<7>({25., 25., 25., 25., 25., 25., 25.},
                                     {.5, .5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_5_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p)
  {
    // return x / y / z / k / m / n / p / q;

    double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta);
    return exp(t1);
  }

  void
  set_true_value()
  {
    true_value = compute_c_zero<7>({10., 10., 10., 10., 10., 10., 10.},
                                   {.5, .5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_6_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double p)
  {
    // return u / v / w / x / y / z / p / t;
    if (u > .9 || v > .8 || w > .7 || x > .6 || y > .5 || z > .4 || p > .3)
      return 0.;
    else
      return exp(10. * u + 9. * v + 8. * w + 7. * x + 6. * y + 5. * z + 4. * p);
  }

  void
  set_true_value()
  {
    true_value = compute_discontinuous<7>({4., 5., 6., 7., 8., 9., 10.},
                                          {.3, .4, .5, .6, .7, .8, .9});
  }

  double true_value;
};

class F_1_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double s, double t, double u, double v, double w, double x)
  {
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x);
  }

  void
  set_true_value()
  {
    true_value = -0.0013062949651908022873;
  }

  double true_value;
};

class F_2_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v, double u)
  {
    // return x / y / z / w / v / u / t / s;

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

  void
  set_true_value()
  {
    true_value = compute_product_peak<6>({50., 50., 50., 50., 50., 50.},
                                         {.5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_3_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v, double u)
  {
    // correct answer: 7.1790160638199853886e-7
    return pow(1. + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -7.);
  }

  void
  set_true_value()
  {
    true_value = 7.1790160638199853886e-7;
  }

  double true_value;
};

class F_4_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v, double u)
  {
    // return x / y / z / w / v / u / t / s;
    double beta = .5;
    return exp(
      -1.0 *
      (pow(25., 2.) * pow(x - beta, 2.) + pow(25., 2.) * pow(y - beta, 2.) +
       pow(25., 2.) * pow(z - beta, 2.) + pow(25., 2.) * pow(w - beta, 2.) +
       pow(25., 2.) * pow(v - beta, 2.) + pow(25., 2.) * pow(u - beta, 2.)));
  }

  void
  set_true_value()
  {
    true_value = compute_gaussian<6>({25., 25., 25., 25., 25., 25.},
                                     {.5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_5_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double k, double m, double n)
  {
    // return x / y / z / k / m / n / p / q;

    double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta);
    return exp(t1);
  }

  void
  set_true_value()
  {
    true_value = compute_c_zero<6>({10., 10., 10., 10., 10., 10.},
                                   {.5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_6_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double y, double v, double x, double w, double z, double u)
  {
    // return u / v / w / x / y / z / p / t;
    if (y > .9 || v > .8 || x > .7 || w > .6 || z > .5 || u > .4)
      return 0.;
    else
      return exp(10. * y + 9. * v + 8. * x + 7. * w + 6. * z + 5. * u);
  }

  void
  set_true_value()
  {
    true_value = compute_discontinuous<6>({5., 6., 7., 8., 9., 10.},
                                          {.4, .5, .6, .7, .8, .9});
  }

  double true_value;
};

class F_1_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double s, double t, double u, double v, double w)
  {
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w);
  }

  void
  set_true_value()
  {
    true_value = 0.020242422119901896863;
  }

  double true_value;
};

class F_2_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v)
  {
    // return x / y / z / w / v / u / t / s;

    const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / pow(a, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(a, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(a, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(a, 2.)) + pow(w - b, 2.));
    const double term_5 = 1. / ((1. / pow(a, 2.)) + pow(v - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5;
    return val;
  }

  void
  set_true_value()
  {
    true_value =
      compute_product_peak<5>({50., 50., 50., 50., 50.}, {.5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_3_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v)
  {
    // return x / y / z / w / v / u / t / s;

    return pow(1. + 5. * v + 4. * w + 3. * x + 2. * y + z, -6.);
  }

  void
  set_true_value()
  {
    true_value = 0.00002602538279621613;
  }

  double true_value;
};

class F_4_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double w, double v)
  {
    // return x / y / z / w / v / u / t / s;
    double beta = .5;
    return exp(
      -1.0 *
      (pow(25., 2.) * pow(x - beta, 2.) + pow(25., 2.) * pow(y - beta, 2.) +
       pow(25., 2.) * pow(z - beta, 2.) + pow(25., 2.) * pow(w - beta, 2.) +
       pow(25., 2.) * pow(v - beta, 2.)));
  }

  void
  set_true_value()
  {
    true_value =
      compute_gaussian<5>({25., 25., 25., 25., 25.}, {.5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_5_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y, double z, double k, double m)
  {
    // return x / y / z / k / m / n / p / q;

    double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta);
    return exp(t1);
  }

  void
  set_true_value()
  {
    true_value =
      compute_c_zero<5>({10., 10., 10., 10., 10.}, {.5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_6_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double y, double x, double w, double v, double u)
  {
    // return u / v / w / x / y / z / p / t;
    if (y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }

  void
  set_true_value()
  {
    true_value =
      compute_discontinuous<5>({5., 6., 7., 8., 9.}, {.4, .5, .6, .7, .8});
  }

  double true_value;
};

#endif
