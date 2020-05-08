#ifndef CUDACUHRE_FUNCTION_CUH
#define CUDACUHRE_FUNCTION_CHH

#define FUN 11
//#define DIM 6
//
#ifndef FUN
#define FUN 2
//#define DIM 3
#endif

#define PI 3.14159265358979323844
#define MIN(a, b) (((a) < (b)) ? a : b)

#define TYPE double

template <typename T>
__device__ T
r8vec_sum(int n, const T a[])
{
  T sum;
  sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum = sum + a[i];
  }
  return sum;
}

template <typename T>
__device__ T
r8vec_dot(int n, T a1[], const T a2[])
{
  int i;
  T value;

  value = 0.0;
  for (i = 0; i < n; i++) {
    value = value + a1[i] * a2[i];
  }
  return value;
}

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
