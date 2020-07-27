#ifndef CUDACUHRE_FUNCTION_CUH
#define CUDACUHRE_FUNCTION_CHH

#include "quad/quad.h"

#define FUN 11

#define PI 3.14159265358979323844
#define MIN(a, b) (((a) < (b)) ? a : b)

class Test {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sin(x + y + z + k + l + m);
  }
};

class FUNC1 {
public:
  // DCUHRE ANSWER with epsrel 1e-4: 2.705514721507 +- 2.70543224E-04
  // range (0,1)

  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    double t1 = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
                pow(m, 2) + pow(n, 2) + pow(o, 2);

    double f = 1.0 / (0.1 + pow(cos(t1), 2));
    return f;
  }
};

class absCosSum5DWithoutK {
  // ESTIMATED ANSWER = 0.6371054
public:
  __device__ __host__ double
  operator()(double v, double w, double x, double y, double z)
  {
    return fabs(cos(4. * v + 5. * w + 6. * x + 7. * y + 8. * z));
  }
};

class absCosSum5D {
  // ESTIMATED ANSWER = 0.6371054
public:
  __device__ __host__ double
  operator()(double v, double w, double x, double y, double z)
  {
    return fabs(cos(4. * v + 5. * w + 6. * x + 7. * y + 8. * z) / 0.6371054);
    // return fabs(cos(4.*v + 5.*w + 6.*x + 7.*y + 8.*z));
  }
};

class BoxIntegral8_15 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {

    double s = 15;
    double sum = 0;
    sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
          pow(m, 2) + pow(n, 2) + pow(o, 2);
    return pow(sum, s / 2);
  }
};

class BoxIntegral8_22 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {

    double s = 22;
    double sum = 0;
    sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
          pow(m, 2) + pow(n, 2) + pow(o, 2);
    return pow(sum, s / 2);
  }
};

class BoxIntegral8_25 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {

    double s = 25;
    double sum = 0;
    sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
          pow(m, 2) + pow(n, 2) + pow(o, 2);
    return pow(sum, s / 2);
  }
};

class BoxIntegral10_15 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o,
             double p,
             double q)
  {

    double s = 15;
    double sum = 0;
    sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
          pow(m, 2) + pow(n, 2) + pow(o, 2) + pow(p, 2) + pow(q, 2);
    return pow(sum, s / 2);
  }
};

class FUNC2 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    // DCUHRE ANSWER with epsrel 1e-4: 0.992581686378 +- 9.92539990E-05
    double xx[8] = {x, y, z, k, l, m, n, o};
    double t1 = 1.0;
    int N = 1;
    int NDIM = 8;
    for (N = 1; N <= NDIM; ++N) {
      t1 = t1 * cos(pow(2.0, 2.0 * N) * xx[N - 1]);
    }

    return cos(t1);
  }
};

class FUNC3 {
public:
  // DCUHRE ANSWER with epsrel 1e-4: 0.991177511809 +- 9.91162890E-05
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    double t1 = 1 * asin(pow(x, 1)) * 2 * asin(pow(y, 2)) * 3 *
                asin(pow(z, 3)) * 4 * asin(pow(k, 4)) * 5 * asin(pow(l, 5)) *
                6 * asin(pow(m, 6)) * 7 * asin(pow(n, 7)) * 8 * asin(pow(o, 8));
    return cos(t1);
  }
};

class FUNC4 {
public:
  // DCUHRE ANSWER with epsrel 1e-4: 0.999020280358 +- 6.69561860E-05
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    double t1 = asin(x) * asin(y) * asin(z) * asin(k) * asin(l) * asin(m) *
                asin(n) * asin(o);
    return cos(t1);
  }
};

class FUNC5 {
public:
  // ANSWER = 4
  // DCUHRE ANSWER with epsrel 1e-4: 4.000009724 +- 0.000395233
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    double sum = 0;
    int NDIM = 8;
    int N = 1;
    double xx[8] = {x, y, z, k, l, m, n, o};
    for (N = 1; N <= NDIM; ++N) {
      sum = sum - cos(10.0 * xx[N - 1]) / 0.054402111088937;
    }
    return sum / 2.0;
  }
};

class FUNC55 {
public:
  // ANSWER = 4
  // DCUHRE ANSWER with epsrel 1e-4: 4.000009724 +- 0.000395233
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    double sum = 0;
    int NDIM = 6;
    int N = 1;
    double xx[6] = {x, y, z, k, l, m};
    for (N = 1; N <= NDIM; ++N) {
      sum = sum - cos(10.0 * xx[N - 1]) / 0.054402111088937;
    }
    return sum / 2.0;
  }
};

class GENZ_1 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    // DCUHRE ANSWER with epsrel 1e-4: -0.731004131572 +- 0.00000000E+00
    int NDIM = 8;
    double alpha = 110.0 / (NDIM * NDIM * sqrt(NDIM * 1.0));
    double sum = pow(alpha, NDIM) * (x + y + z + k + l + m + n + o);
    sum += 2.0 * PI * alpha;
    return cos(sum);
  }
};

class GENZ_2 {
public:
  // DCUHRE ANSWER with epsrel 1e-4: 0.329102695990 +- 3.16149770E-05
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double p)
  {
    int NDIM = 8;
    double alpha = 600.0 / (NDIM * NDIM * NDIM);
    double t1 = 1.0 / pow(alpha, 2);
    double total =
      (1.0 / (t1 + pow(x - alpha, 2))) * (1.0 / (t1 + pow(y - alpha, 2))) *
      (1.0 / (t1 + pow(z - alpha, 2))) * (1.0 / (t1 + pow(k - alpha, 2))) *
      (1.0 / (t1 + pow(l - alpha, 2))) * (1.0 / (t1 + pow(m - alpha, 2))) *
      (1.0 / (t1 + pow(n - alpha, 2))) * (1.0 / (t1 + pow(p - alpha, 2)));
    return total;
  }
};

class GENZ_4 {
public:
  // DCUHRE ANSWER with epsrel 1e-4: 0.720492998631 +- 5.00191000E-07
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    double alpha = 0.5;
    double beta = alpha;
    double t1 = pow(x - beta, 2) + pow(y - beta, 2) + pow(z - beta, 2) +
                pow(k - beta, 2) + pow(l - beta, 2) + pow(m - beta, 2) +
                pow(n - beta, 2) + pow(o - beta, 2);
    return exp(-1 * t1 * alpha);
  }
};

class GENZ_5 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    // DCUHRE ANSWER with epsrel 1e-4: 0.375625473524 +- 3.69126870E-05
    double alpha = 0.5;
    double beta = alpha;
    // int N = 1;

    double t1 = fabs(x - beta) + fabs(y - beta) + fabs(z - beta) +
                fabs(k - beta) + fabs(l - beta) + fabs(m - beta) +
                fabs(n - beta) + fabs(o - beta);
    return exp(-1 * t1 * alpha);
  }
};

template <typename T>
__device__ T
IntegrandFunc(const T xx[], int NDIM)
{
  T f = 0;
#if FUN == 1
  T t1 = 0;
  int N = 1;
  for (N = 1; N <= NDIM; ++N) {
    t1 += pow(xx[N - 1], 2);
  }
  f = 1.0 / (0.1 + pow(cos(t1), 2));
#elif FUN == 2
  // FTEST3
  T t1 = 1.0;
  int N = 1;
  for (N = 1; N <= NDIM; ++N) {
    t1 = t1 * cos(pow(2.0, 2.0 * N) * xx[N - 1]);
  }
  f = cos(t1);
#elif FUN == 3
  // FTEST6
  T t1 = 1.0;
  int N = 1;
  for (N = 1; N <= NDIM; ++N) {
    t1 = t1 * N * asin(pow(xx[N - 1], N));
  }
  f = sin(t1);
#elif FUN == 4
  // FTEST7
  T t1 = 1.0;
  int N = 1;
  for (N = 1; N <= NDIM; ++N) {
    t1 = t1 * asin(xx[N - 1]);
  }
  f = sin(t1);
#elif FUN == 5
  T sum = 0;
  int N = 1;
  for (N = 1; N <= NDIM; ++N) {
    sum = sum - cos(10.0 * xx[N - 1]) / 0.054402111088937;
  }
  f = sum / 2.0;
#elif FUN == 6
  T sum = 0;
  int N = 1;
  T alpha = 110.0 / (NDIM * NDIM * sqrt(NDIM * 1.0));
  for (N = 1; N <= NDIM; ++N) {
    sum += alpha * xx[N - 1];
  }
  sum = 2.0 * PI * alpha + sum;
  f = cos(sum);
#elif FUN == 7
  int N = 1;
  T total = 1.0;
  T alpha = 600.0 / (NDIM * NDIM * NDIM);
  T beta = alpha;
  for (N = 1; N <= NDIM; ++N) {
    total = total * (1.0 / pow(alpha, 2) + pow(xx[N - 1] - beta, 2));
  }
  f = 1.0 / total;

#elif FUN == 10
  T total = 0.0;
  T alpha = 0.5; // 150.0/(NDIM * NDIM * NDIM);
  T beta = alpha;
  int N = 1;
  for (N = 1; N <= NDIM; ++N) {
    total = total + alpha * fabs(xx[N - 1] - beta);
  }
  f = exp(-total);

#elif FUN == 11
  // sin(x1+x2+x3+x4+x5+x6) from 0 to 10
  double sum = 0;
  int N = 0;
  for (N = 0; N < NDIM; N++)
    sum += 10 * xx[N];

  f = pow(10, NDIM) * sin(sum);

#elif FUN == 12
  // sin(x1+x2+x3+x4+x5+x6)+1 from 0 to 1
  double sum = 0;
  int N = 0;
  for (N = 0; N < NDIM; N++)
    sum += xx[N];

  f = 1 + sin(sum);
#elif FUN == 13
  // sin(x1+x2+x3+x4+x5+x6)+1 from 0 to 10
  double sum = 0;
  int N = 0;
  for (N = 0; N < NDIM; N++)
    sum += 10 * xx[N];

  f = pow(10, NDIM) * (1 + sin(sum));
#elif FUN == 14
  // sin(10[x1+x2+x3+x4+x5+x6]) from 0 to 1
  double sum = 0;
  int N = 0;
  for (N = 0; N < NDIM; N++)
    sum += xx[N];

  f = sin(10 * sum);

#elif FUN == 15
  // this is FUN 14 in demo2.c
  double sum = 0;

  int N = 0;
  for (N = 0; N < NDIM; N++) {
    if (N == 0)
      sum += 5 * xx[N];
    else if (N == 1)
      sum += .001 * xx[N];
    else if (N == 2)
      sum += xx[N];
    else
      sum += xx[N] * .001;
  }

  f = sin(sum);
#elif FUN == 19
  // Sin (x1 - x2 + x3 - x4 + x5 - x6) in (0,1)
  double sum = 0;
  sum = xx[0] - xx[1] + xx[2] - xx[3] + xx[4] - xx[5];
  f = sin(sum);
#elif FUN == 20
  // Sin (x1 - x2 + x3 - x4 + x5 - x6) in (0,10)
  double sum = 0;
  sum = 10 * (xx[0] - xx[1] + xx[2] - xx[3] + xx[4] - xx[5]);
  f = pow(10, 6) * sin(sum);
#endif

  return f;
}

#endif
