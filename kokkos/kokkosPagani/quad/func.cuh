#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979323844

__inline__ __device__ double
func0(double x[], int ndim)
{
  // printf("within function\n");
  double value = 0.;
  for (int i = 1; i <= ndim; i++) {
    value += x[i];
  }
  return (sin(value));
}

__inline__ double
func1(double x[], int ndim)
{
  double value = 0.;
  // double PI = 3.14159265358979323844;
  for (int i = 1; i <= ndim; i++) {
    value = pow(x[i], 2.0);
  }
  return exp(-1 * value / (2 * pow(0.01, 2))) *
         (1 / pow(sqrt(2 * PI) * 0.01, 9));
}

__inline__ double
func2(double x[], int ndim)
{
  double value;
  value = 1.0;
  for (int i = 1; i <= ndim; i++) {
    value += (ndim + 1 - i) * x[i];
  }
  value = pow(value, (ndim + 1));
  value = 1 / value;
  return (value);
}

__inline__ double
func3(double x[], int ndim)
{
  double sigma = 0.31622776601683794;
  // double sigma = 0.02;
  double k;
  int j;
  k = (sigma * sqrt(2.0 * M_PI));
  k = pow(k, 9);
  k = 1.0 / k;
  // int ndim = 9;
  double tsum = 0.0;
  for (j = 1; j <= ndim; j++) {
    tsum += (x[j]) * (x[j]);
  }
  tsum = -tsum / (2.0 * sigma * sigma);
  tsum = exp(tsum);
  return (tsum * k);
}