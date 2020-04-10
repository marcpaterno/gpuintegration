#define FUN 5
#define DIM 8

#ifndef FUN
#define FUN 2
#define DIM 3
#endif


#define PI 3.14159265358979323844
#define MIN(a, b) (((a) < (b))?a:b)

#define TYPE double

template<typename T>
__device__
T r8vec_sum ( int n, const T a[] ){
  T sum;
  sum = 0.0;
  for (int i = 0; i < n; i++ ){
    sum = sum + a[i];
  }
  return sum;
}

template<typename T>
__device__
T r8vec_dot ( int n, T a1[], const T a2[] ){
  int i;
  T value;

  value = 0.0;
  for ( i = 0; i < n; i++ ){
    value = value + a1[i] * a2[i];
  }
  return value;
}

template<typename T>
__device__
T IntegrandFunc(const T xx[], int NDIM) {
  T f = 0;

#if FUN == 1
  T t1 = 0;
  int N =1;
  for(N = 1; N <= NDIM; ++N){
    t1 += pow(xx[N-1], 2);
  }
  f = 1.0/(0.1 + pow(cos(t1),2));
#elif FUN == 2
  //FTEST3
  T t1 = 1.0;
  int N = 1;
  for(N=1; N <= NDIM; ++N){
    t1 = t1 * cos(pow(2.0, 2.0 * N) * xx[N-1]);
  }
  f = cos(t1);
#elif FUN == 3
  //FTEST6
  T t1 = 1.0;
  int N = 1;
  for(N = 1; N <= NDIM; ++N){
    t1 = t1 * N * asin(pow(xx[N-1],N));
  }
  f = sin(t1);
#elif FUN == 4
  //FTEST7
  T t1 = 1.0;
  int N = 1;
  for(N = 1; N <= NDIM; ++N){
    t1 = t1 * asin(xx[N-1]);
  }
  f = sin(t1);
#elif FUN == 5
  T sum = 0;
  int N = 1;
  for(N = 1; N <= NDIM; ++N){
   sum = sum - cos(10.0 * xx[N-1])/0.054402111088937;
  }
  f = sum/2.0;
#elif FUN == 6
  T sum = 0;
  int N = 1;
  T alpha = 110.0/(NDIM * NDIM * sqrt(NDIM*1.0));
  for(N = 1; N <= NDIM; ++N){
    sum += alpha * xx[N-1];
  }
  sum = 2.0 * PI * alpha + sum;
  f = cos (sum);
#elif FUN == 7
  int N = 1;
  T total = 1.0;
  T alpha = 600.0/(NDIM * NDIM * NDIM);
  T beta = alpha;
  for(N = 1; N <= NDIM; ++N){
    total = total * (1.0 / pow ( alpha, 2) + pow ( xx[N-1] - beta, 2 ) );
  }
  f = 1.0 / total;

#elif FUN == 10
  T total = 0.0;
  T alpha = 0.5;//150.0/(NDIM * NDIM * NDIM);
  T beta = alpha;
  int N = 1;
  for(N = 1; N <= NDIM; ++N){
    total = total + alpha * fabs ( xx[N-1] - beta);
  }
  f = exp ( - total );
  
#endif
  
  return f;
}




  
