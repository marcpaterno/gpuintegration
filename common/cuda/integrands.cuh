#ifndef CUDA_INTREGRANDS_CUH
#define CUDA_INTEGRANDS_CUH

#include "cuda/pagani/demos/compute_genz_integrals.cuh"

// We have no analytic solution for F3 integrands with mathematica or the Genz
// testpack (corner-peak integrand seems slighlty different there) same thing
// applies for the F1 oscillatory integrand That's why for F1 and F3 we use
// mathematica to get the exact solution to the integrand with the particular
// set of coefficients and parameters

class F_5_8D_alt {
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
    // return x / y / z / k / m / n / p / q;

    double beta = .5;
    double t1 = -sharpness * fabs(x - beta) - sharpness * fabs(y - beta) -
                sharpness * fabs(z - beta) - sharpness * fabs(k - beta) -
                sharpness * fabs(m - beta) - sharpness * fabs(n - beta) -
                sharpness * fabs(p - beta) - sharpness * fabs(q - beta);
    return exp(t1);
  }

  void
  set_true_value()
  {
    true_value = compute_c_zero<8>({sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness},
                                   {.5, .5, .5, .5, .5, .5, .5, .5});
  }

  double sharpness = 0;
  double true_value = 0;
};

class F_5_7D_alt {
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
    // return x / y / z / k / m / n / p / q;

    double beta = .5;
    double t1 = -sharpness * fabs(x - beta) - sharpness * fabs(y - beta) -
                sharpness * fabs(z - beta) - sharpness * fabs(k - beta) -
                sharpness * fabs(m - beta) - sharpness * fabs(n - beta) -
                sharpness * fabs(p - beta);
    return exp(t1);
  }

  void
  set_true_value()
  {
    true_value = compute_c_zero<7>({sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness,
                                    sharpness},
                                   {.5, .5, .5, .5, .5, .5, .5});
  }

  double sharpness = 0;
  double true_value = 0;
};

class F_5_6D_alt {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double m, double n)
  {
    // return x / y / z / k / m / n / p / q;

    double beta = .5;
    double t1 = -sharpness * fabs(x - beta) - sharpness * fabs(y - beta) -
                sharpness * fabs(z - beta) - sharpness * fabs(k - beta) -
                sharpness * fabs(m - beta) - sharpness * fabs(n - beta);
    return exp(t1);
  }

  void
  set_true_value()
  {
    true_value = compute_c_zero<6>(
      {sharpness, sharpness, sharpness, sharpness, sharpness, sharpness},
      {.5, .5, .5, .5, .5, .5});
  }

  double sharpness = 0;
  double true_value = 0;
};

class F_5_5D_alt {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double m)
  {
    // return x / y / z / k / m / n / p / q;

    double beta = .5;
    double t1 = -sharpness * fabs(x - beta) - sharpness * fabs(y - beta) -
                sharpness * fabs(z - beta) - sharpness * fabs(k - beta) -
                sharpness * fabs(m - beta);
    return exp(t1);
  }

  void
  set_true_value()
  {
    true_value =
      compute_c_zero<5>({sharpness, sharpness, sharpness, sharpness, sharpness},
                        {.5, .5, .5, .5, .5});
  }

  double sharpness = 0;
  double true_value = 0;
};

class F_4_8D_alt {
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
    // return x / y / z / w / v / u / t / s;
    return exp(-1.0 * (pow(sharpness, 2.) * pow(x - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(y - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(z - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(w - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(v - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(u - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(t - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(s - peak_loc, 2.)));
  }

  double sharpness = 0;
  double peak_loc = 0;

  void
  set_true_value()
  {
    true_value = compute_gaussian<8>({sharpness,
                                      sharpness,
                                      sharpness,
                                      sharpness,
                                      sharpness,
                                      sharpness,
                                      sharpness,
                                      sharpness},
                                     {peak_loc,
                                      peak_loc,
                                      peak_loc,
                                      peak_loc,
                                      peak_loc,
                                      peak_loc,
                                      peak_loc,
                                      peak_loc});
  }

  double true_value;
};

class F_4_7D_alt {
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
    // return x / y / z / w / v / u / t / s;
    return exp(-1.0 * (pow(sharpness, 2.) * pow(x - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(y - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(z - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(w - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(v - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(u - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(t - peak_loc, 2.)));
  }

  double sharpness = 0;
  double peak_loc = 0;
  void
  set_true_value()
  {
    true_value = compute_gaussian<7>(
      {sharpness,
       sharpness,
       sharpness,
       sharpness,
       sharpness,
       sharpness,
       sharpness},
      {peak_loc, peak_loc, peak_loc, peak_loc, peak_loc, peak_loc, peak_loc});
  }

  double true_value;
};

class F_4_6D_alt {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v, double u)
  {
    // return x / y / z / w / v / u / t / s;
    return exp(-1.0 * (pow(sharpness, 2.) * pow(x - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(y - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(z - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(w - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(v - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(u - peak_loc, 2.)));
  }

  double sharpness = 0;
  double peak_loc = 0;

  void
  set_true_value()
  {
    true_value = compute_gaussian<6>(
      {sharpness, sharpness, sharpness, sharpness, sharpness, sharpness},
      {peak_loc, peak_loc, peak_loc, peak_loc, peak_loc, peak_loc});
  }

  double true_value;
};

class F_4_5D_alt {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v)
  {
    // return x / y / z / w / v / u / t / s;
    return exp(-1.0 * (pow(sharpness, 2.) * pow(x - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(y - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(z - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(w - peak_loc, 2.) +
                       pow(sharpness, 2.) * pow(v - peak_loc, 2.)));
  }

  double sharpness = 0;
  double peak_loc = 0;

  void
  set_true_value()
  {
    true_value = compute_gaussian<5>(
      {sharpness, sharpness, sharpness, sharpness, sharpness},
      {peak_loc, peak_loc, peak_loc, peak_loc, peak_loc});
  }

  double true_value;
};

class F_6_8D_alt {
public:
  __device__ __host__ double
  operator()(double t,
             double p,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
  {
    if (z > disc_bound || y > disc_bound || x > disc_bound || w > disc_bound ||
        v > disc_bound || u > disc_bound || p > disc_bound || t > disc_bound)
      return 0.;
    else
      return exp(sharpness * z + sharpness * y + sharpness * x + sharpness * w +
                 sharpness * v + sharpness * u + sharpness * p + sharpness * t);
  }

  void
  set_true_value()
  {
    true_value = compute_discontinuous<8>({sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness},
                                          {disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound});
  }

  double sharpness = 0;
  double disc_bound = 0;
  double true_value;
};

class F_6_7D_alt {
public:
  __device__ __host__ double
  operator()(double t,
             double p,
             double u,
             double v,
             double w,
             double x,
             double y)
  {
    if (y > disc_bound || x > disc_bound || w > disc_bound || v > disc_bound ||
        u > disc_bound || p > disc_bound || t > disc_bound)
      return 0.;
    else
      return exp(sharpness * y + sharpness * x + sharpness * w + sharpness * v +
                 sharpness * u + sharpness * p + sharpness * t);
  }

  void
  set_true_value()
  {
    true_value = compute_discontinuous<7>({sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness,
                                           sharpness},
                                          {disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound,
                                           disc_bound});
  }

  double sharpness = 0;
  double disc_bound = 0;
  double true_value;
};

class F_6_6D_alt {
public:
  __device__ __host__ double
  operator()(double t, double p, double u, double v, double w, double x)
  {
    if (x > disc_bound || w > disc_bound || v > disc_bound || u > disc_bound ||
        p > disc_bound || t > disc_bound)
      return 0.;
    else
      return exp(sharpness * x + sharpness * w + sharpness * v + sharpness * u +
                 sharpness * p + sharpness * t);
  }

  void
  set_true_value()
  {
    true_value = compute_discontinuous<6>(
      {sharpness, sharpness, sharpness, sharpness, sharpness, sharpness},
      {disc_bound, disc_bound, disc_bound, disc_bound, disc_bound, disc_bound});
  }

  double sharpness = 0;
  double disc_bound = 0;
  double true_value;
};

class F_6_5D_alt {
public:
  __device__ __host__ double
  operator()(double t, double p, double u, double v, double w)
  {
    if (w > disc_bound || v > disc_bound || u > disc_bound || p > disc_bound ||
        t > disc_bound)
      return 0.;
    else
      return exp(sharpness * w + sharpness * v + sharpness * u + sharpness * p +
                 sharpness * t);
  }

  void
  set_true_value()
  {
    true_value = compute_discontinuous<5>(
      {sharpness, sharpness, sharpness, sharpness, sharpness},
      {disc_bound, disc_bound, disc_bound, disc_bound, disc_bound});
  }

  double sharpness = 0;
  double disc_bound = 0;
  double true_value;
};

class F_2_8D_alt {
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
    const double b = .5;
    const double term_1 = 1. / ((1. / pow(alpha, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(alpha, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(alpha, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(alpha, 2.)) + pow(w - b, 2.));
    const double term_5 = 1. / ((1. / pow(alpha, 2.)) + pow(v - b, 2.));
    const double term_6 = 1. / ((1. / pow(alpha, 2.)) + pow(u - b, 2.));
    const double term_7 = 1. / ((1. / pow(alpha, 2.)) + pow(t - b, 2.));
    const double term_8 = 1. / ((1. / pow(alpha, 2.)) + pow(s - b, 2.));

    double val =
      term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7 * term_8;
    return val;
  }

  double alpha = 0;

  void
  set_true_value()
  {
    true_value = compute_product_peak<8>(
      {alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha},
      {.5, .5, .5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_2_7D_alt {
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
    const double b = .5;
    const double term_1 = 1. / ((1. / pow(alpha, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(alpha, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(alpha, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(alpha, 2.)) + pow(w - b, 2.));
    const double term_5 = 1. / ((1. / pow(alpha, 2.)) + pow(v - b, 2.));
    const double term_6 = 1. / ((1. / pow(alpha, 2.)) + pow(u - b, 2.));
    const double term_7 = 1. / ((1. / pow(alpha, 2.)) + pow(t - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7;
    return val;
  }

  double alpha = 0;

  void
  set_true_value()
  {
    true_value =
      compute_product_peak<7>({alpha, alpha, alpha, alpha, alpha, alpha, alpha},
                              {.5, .5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_2_6D_alt {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v, double u)
  {
    const double b = .5;
    const double term_1 = 1. / ((1. / pow(alpha, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(alpha, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(alpha, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(alpha, 2.)) + pow(w - b, 2.));
    const double term_5 = 1. / ((1. / pow(alpha, 2.)) + pow(v - b, 2.));
    const double term_6 = 1. / ((1. / pow(alpha, 2.)) + pow(u - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }

  double alpha = 0;

  void
  set_true_value()
  {
    true_value = compute_product_peak<6>(
      {alpha, alpha, alpha, alpha, alpha, alpha}, {.5, .5, .5, .5, .5, .5});
  }

  double true_value;
};

class F_2_5D_alt {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v)
  {
    const double b = .5;
    const double term_1 = 1. / ((1. / pow(alpha, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(alpha, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(alpha, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(alpha, 2.)) + pow(w - b, 2.));
    const double term_5 = 1. / ((1. / pow(alpha, 2.)) + pow(v - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5;
    return val;
  }

  double alpha = 0;

  void
  set_true_value()
  {
    true_value = compute_product_peak<5>({alpha, alpha, alpha, alpha, alpha},
                                         {.5, .5, .5, .5, .5});
  }

  double true_value;
};

class G_func_10D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q,
             double r,
             double t)
  {
    double term1 = (std::abs(4 * x - 2) + (1 - 2) / 2) / (1 + (1 - 2) / 2);
    double term2 = (std::abs(4 * y - 2) + (2 - 2) / 2) / (1 + (2 - 2) / 2);
    double term3 = (std::abs(4 * z - 2) + (3 - 2) / 2) / (1 + (3 - 2) / 2);
    double term4 = (std::abs(4 * k - 2) + (4 - 2) / 2) / (1 + (4 - 2) / 2);
    double term5 = (std::abs(4 * m - 2) + (5 - 2) / 2) / (1 + (5 - 2) / 2);
    double term6 = (std::abs(4 * n - 2) + (6 - 2) / 2) / (1 + (6 - 2) / 2);
    double term7 = (std::abs(4 * p - 2) + (7 - 2) / 2) / (1 + (7 - 2) / 2);
    double term8 = (std::abs(4 * q - 2) + (8 - 2) / 2) / (1 + (8 - 2) / 2);
    double term9 = (std::abs(4 * r - 2) + (9 - 2) / 2) / (1 + (9 - 2) / 2);
    double term10 = (std::abs(4 * t - 2) + (10 - 2) / 2) / (1 + (10 - 2) / 2);

    return term1 * term2 * term3 * term4 * term5 * term6 * term7 * term8 *
           term9 * term10;
  }

  void
  set_true_value(double low, double high)
  {
    true_value = 1.;
  }

  double true_value = 0;
};

class G_func_9D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q,
             double r)
  {
    double term1 = (std::abs(4 * x - 2) + (1 - 2) / 2) / (1 + (1 - 2) / 2);
    double term2 = (std::abs(4 * y - 2) + (2 - 2) / 2) / (1 + (2 - 2) / 2);
    double term3 = (std::abs(4 * z - 2) + (3 - 2) / 2) / (1 + (3 - 2) / 2);
    double term4 = (std::abs(4 * k - 2) + (4 - 2) / 2) / (1 + (4 - 2) / 2);
    double term5 = (std::abs(4 * m - 2) + (5 - 2) / 2) / (1 + (5 - 2) / 2);
    double term6 = (std::abs(4 * n - 2) + (6 - 2) / 2) / (1 + (6 - 2) / 2);
    double term7 = (std::abs(4 * p - 2) + (7 - 2) / 2) / (1 + (7 - 2) / 2);
    double term8 = (std::abs(4 * q - 2) + (8 - 2) / 2) / (1 + (8 - 2) / 2);
    double term9 = (std::abs(4 * r - 2) + (9 - 2) / 2) / (1 + (9 - 2) / 2);
    return term1 * term2 * term3 * term4 * term5 * term6 * term7 * term8 *
           term9;
  }

  void
  set_true_value(double low, double high)
  {
    true_value = 1.;
  }

  double true_value = 0;
};

class G_func_8D {
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
    double term1 = (std::abs(4 * x - 2) + (1 - 2) / 2) / (1 + (1 - 2) / 2);
    double term2 = (std::abs(4 * y - 2) + (2 - 2) / 2) / (1 + (2 - 2) / 2);
    double term3 = (std::abs(4 * z - 2) + (3 - 2) / 2) / (1 + (3 - 2) / 2);
    double term4 = (std::abs(4 * k - 2) + (4 - 2) / 2) / (1 + (4 - 2) / 2);
    double term5 = (std::abs(4 * m - 2) + (5 - 2) / 2) / (1 + (5 - 2) / 2);
    double term6 = (std::abs(4 * n - 2) + (6 - 2) / 2) / (1 + (6 - 2) / 2);
    double term7 = (std::abs(4 * p - 2) + (7 - 2) / 2) / (1 + (7 - 2) / 2);
    double term8 = (std::abs(4 * q - 2) + (8 - 2) / 2) / (1 + (8 - 2) / 2);
    return term1 * term2 * term3 * term4 * term5 * term6 * term7 * term8;
  }

  void
  set_true_value(double low, double high)
  {
    true_value = 1.;
  }

  double true_value = 0;
};

class G_func_7D {
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
    double term1 = (std::abs(4 * x - 2) + (1 - 2) / 2) / (1 + (1 - 2) / 2);
    double term2 = (std::abs(4 * y - 2) + (2 - 2) / 2) / (1 + (2 - 2) / 2);
    double term3 = (std::abs(4 * z - 2) + (3 - 2) / 2) / (1 + (3 - 2) / 2);
    double term4 = (std::abs(4 * k - 2) + (4 - 2) / 2) / (1 + (4 - 2) / 2);
    double term5 = (std::abs(4 * m - 2) + (5 - 2) / 2) / (1 + (5 - 2) / 2);
    double term6 = (std::abs(4 * n - 2) + (6 - 2) / 2) / (1 + (6 - 2) / 2);
    double term7 = (std::abs(4 * p - 2) + (7 - 2) / 2) / (1 + (7 - 2) / 2);
    return term1 * term2 * term3 * term4 * term5 * term6 * term7;
  }

  void
  set_true_value(double low, double high)
  {
    true_value = 1.;
  }

  double true_value = 0;
};

class G_func_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double m, double n)
  {
    double term1 = (std::abs(4 * x - 2) + (1 - 2) / 2) / (1 + (1 - 2) / 2);
    double term2 = (std::abs(4 * y - 2) + (2 - 2) / 2) / (1 + (2 - 2) / 2);
    double term3 = (std::abs(4 * z - 2) + (3 - 2) / 2) / (1 + (3 - 2) / 2);
    double term4 = (std::abs(4 * k - 2) + (4 - 2) / 2) / (1 + (4 - 2) / 2);
    double term5 = (std::abs(4 * m - 2) + (5 - 2) / 2) / (1 + (5 - 2) / 2);
    double term6 = (std::abs(4 * n - 2) + (6 - 2) / 2) / (1 + (6 - 2) / 2);
    return term1 * term2 * term3 * term4 * term5 * term6;
  }

  void
  set_true_value(double low, double high)
  {
    true_value = 1.;
  }

  double true_value = 0;
};

class G_func_5D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double m)
  {
    double term1 = (std::abs(4 * x - 2) + (1 - 2) / 2) / (1 + (1 - 2) / 2);
    double term2 = (std::abs(4 * y - 2) + (2 - 2) / 2) / (1 + (2 - 2) / 2);
    double term3 = (std::abs(4 * z - 2) + (3 - 2) / 2) / (1 + (3 - 2) / 2);
    double term4 = (std::abs(4 * k - 2) + (4 - 2) / 2) / (1 + (4 - 2) / 2);
    double term5 = (std::abs(4 * m - 2) + (5 - 2) / 2) / (1 + (5 - 2) / 2);

    return term1 * term2 * term3 * term4 * term5;
  }

  void
  set_true_value(double low, double high)
  {
    true_value = 1.;
  }

  double true_value = 0;
};

class Cos_fully_sep_product_1D {
public:
  __device__ __host__ double
  operator()(double x)
  {
    return cos(x);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<1>(low, high);
  }

  double true_value = 0;
};

class Cos_fully_sep_product_2D {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    return cos(x) * cos(y);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<2>(low, high);
  }

  double true_value = 0;
};

class Cos_fully_sep_product_4D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k)
  {
    return cos(x) * cos(y) * cos(z) * cos(k);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<4>(low, high);
  }

  double true_value = 0;
};

class Cos_fully_sep_product_5D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double m)
  {
    return cos(x) * cos(y) * cos(z) * cos(k) * cos(m);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<5>(low, high);
  }

  double true_value = 0;
};

class Cos_fully_sep_product_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double m, double n)
  {
    return cos(x) * cos(y) * cos(z) * cos(k) * cos(m) * cos(n);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<6>(low, high);
  }

  double true_value = 0;
};

class Cos_fully_sep_product_7D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double r)
  {
    return cos(x) * cos(y) * cos(z) * cos(k) * cos(m) * cos(n) * cos(r);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<7>(low, high);
  }

  double true_value = 0;
};

class Cos_fully_sep_product_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double r,
             double d)
  {
    return cos(x) * cos(y) * cos(z) * cos(k) * cos(m) * cos(n) * cos(r) *
           cos(d);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<8>(low, high);
  }

  double true_value = 0;
};

class Cos_fully_sep_product_9D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double r,
             double d,
             double p)
  {
    return cos(x) * cos(y) * cos(z) * cos(k) * cos(m) * cos(n) * cos(r) *
           cos(d) * cos(p);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<9>(low, high);
  }

  double true_value = 0;
};

class Cos_fully_sep_product_10D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double r,
             double d,
             double p,
             double t)
  {
    return cos(x) * cos(y) * cos(z) * cos(k) * cos(m) * cos(n) * cos(r) *
           cos(d) * cos(p) * cos(t);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_fully_sep_product_at_bounds<10>(low, high);
  }

  double true_value = 0;
};

class Cos_semi_sep_product_4D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k)
  {
    return cos(x + y) * cos(z + k);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_semi_sep_product_at_bounds<4>(low, high);
  }

  double true_value = 0;
};

class Cos_semi_sep_product_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double m, double n)
  {
    return cos(x + y) * cos(z + k) * cos(m + n);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_semi_sep_product_at_bounds<6>(low, high);
  }

  double true_value = 0;
};

class Cos_semi_sep_product_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double r,
             double d)
  {
    return cos(x + y) * cos(z + k) * cos(m + n) * cos(r + d);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_semi_sep_product_at_bounds<8>(low, high);
  }

  double true_value = 0;
};

class Oscillatory_2D {
public:
  __device__ __host__ double
  operator()(double s, double t)
  {
    return cos(s + t);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<2>(low, high);
  }

  double true_value;
};

class Oscillatory_3D {
public:
  __device__ __host__ double
  operator()(double s, double t, double u)
  {
    return cos(s + t + u);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<3>(low, high);
  }

  double true_value = 0.;
};

class Oscillatory_4D {
public:
  __device__ __host__ double
  operator()(double s, double t, double u, double v)
  {
    return cos(s + t + u + v);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<4>(low, high);
  }

  double true_value = 0.;
};

class Oscillatory_5D {
public:
  __device__ __host__ double
  operator()(double s, double t, double u, double v, double w)
  {
    return cos(s + t + u + v + w);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<5>(low, high);
  }

  double true_value = 0.;
};

class Oscillatory_6D {
public:
  __device__ __host__ double
  operator()(double s, double t, double u, double v, double w, double x)
  {
    return cos(s + t + u + v + w + x);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<6>(low, high);
  }

  double true_value = 0.;
};

class Oscillatory_7D {
public:
  __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y)
  {
    return cos(s + t + u + v + w + x + y);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<7>(low, high);
  }

  double true_value = 0.;
};

class Oscillatory_8D {
public:
  __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
  {
    return cos(s + t + u + v + w + x + y + z);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<8>(low, high);
  }

  double true_value = 0.;
};

class Oscillatory_9D {
public:
  __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double c)
  {
    return cos(s + t + u + v + w + x + y + z + c);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<9>(low, high);
  }

  double true_value = 0.;
};

class Oscillatory_10D {
public:
  __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double c,
             double b)
  {
    return cos(s + t + u + v + w + x + y + z + c + b);
  }

  void
  set_true_value(double low, double high)
  {
    true_value = compute_cos_non_sep_product_at_bounds<10>(low, high);
  }
  double true_value = 0.;
};

class Addition_8D {
public:
  __host__ __device__ double
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
  __host__ __device__ double
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
  __host__ __device__ double
  operator()(double x, double y, double z, double w, double v, double m)
  {
    return x + y + z + w + v + m;
  }
};

class Addition_5D {
public:
  __host__ __device__ double
  operator()(double x, double y, double z, double w, double v)
  {
    return x + y + z + w + v;
  }
};

class Addition_4D {
public:
  __host__ __device__ double
  operator()(double x, double y, double z, double w)
  {
    return x + y + z + w;
  }
};

class Addition_3D {
public:
  __host__ __device__ double
  operator()(double x, double y, double z)
  {
    return x + y + z;
  }
};

class SinSum_3D {
public:
  __host__ __device__ double
  operator()(double x, double y, double z)
  {
    return sin(x + y + z);
  }
};

class SinSum_4D {
public:
  __host__ __device__ double
  operator()(double x, double y, double z, double k)
  {
    return sin(x + y + z + k);
  }
};

class SinSum_5D {
public:
  __host__ __device__ double
  operator()(double x, double y, double z, double k, double l)
  {
    return sin(x + y + z + k + l);
  }
};

class SinSum_6D {
public:
  __host__ __device__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

class SinSum_7D {
public:
  __host__ __device__ double
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
  __host__ __device__ double
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
  __host__ __device__ double
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
  __device__ __host__ double
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
  __host__ __device__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __host__ __device__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __host__ __device__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
  __device__ __host__ double
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
