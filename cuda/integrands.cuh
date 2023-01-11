#ifndef CUDA_INTREGRANDS_CUH
#define CUDA_INTEGRANDS_CUH

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
  operator()(double x, double y, double z, double k, double l, double m, double n)
  {
    return sin(x + y + z + k + l + m + n);
  }
};

class SinSum_8D {
public:
  __host__ __device__ double
  operator()(double x, double y, double z, double k, double l, double m, double n, double p)
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

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7 * term_8;
    return val;
  }
};

class F_3_8D{
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
	return pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
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
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.) + 
				pow(25., 2.) * pow(u - beta, 2.) + 
				pow(25., 2.) * pow(t - beta, 2.) + 
				pow(25., 2.) * pow(s - beta, 2.)));
    }
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
	//return x / y / z / k / m / n / p / q;
  
	double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta) - 10. * fabs(q - beta);
    return exp(t1);
  }
};

class F_6_8D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	//return u / v / w / x / y / z / p / t;
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3 || t > .2)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. *p + 3. * t);
  }
};

class F_1_6D {
public:
  __host__ __device__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x)
  {
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x);
  }
};

class F_2_6D {
public:
  __device__ __host__ double
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
    const double term_1 = 1. / ((1. / pow(a, 2.)) + pow(x - b, 2.));
    const double term_2 = 1. / ((1. / pow(a, 2.)) + pow(y - b, 2.));
    const double term_3 = 1. / ((1. / pow(a, 2.)) + pow(z - b, 2.));
    const double term_4 = 1. / ((1. / pow(a, 2.)) + pow(w - b, 2.));
    const double term_5 = 1. / ((1. / pow(a, 2.)) + pow(v - b, 2.));
    const double term_6 = 1. / ((1. / pow(a, 2.)) + pow(u - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

class F_3_6D{
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
  {
	//return x / y / z / w / v / u / t / s;
  
	return pow(1. + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_6D {
  public:
    __device__ __host__ double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
    {
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.) + 
				pow(25., 2.) * pow(u - beta, 2.)));
    }
};

class F_5_6D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n)
  {
	//return x / y / z / k / m / n / p / q;
  
	double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta);
    return exp(t1);
  }
};

class F_6_6D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z)
  {
	//return u / v / w / x / y / z / p / t;
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }
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
	//return x / y / z / w / v / u / t / s;
  
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
};

class F_3_7D{
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
	//return x / y / z / w / v / u / t / s;
  
	return pow(1. + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
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
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.) + 
				pow(25., 2.) * pow(u - beta, 2.) + 
				pow(25., 2.) * pow(t - beta, 2.)));
    }
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
	//return x / y / z / k / m / n / p / q;
  
	double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta);
    return exp(t1);
  }
};

class F_6_7D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z, double p)
  {
	//return u / v / w / x / y / z / p / t;
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3)
      return 0.;
    else
      return exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. *p);
  }
};

class F_1_5D {
public:
  __host__ __device__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w)
  {
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w);
  }
};

class F_2_5D {
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v)
  {
	//return x / y / z / w / v / u / t / s;
  
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
};

class F_3_5D{
public:
  __device__ __host__ double
  operator()(double x,
               double y,
               double z,
               double w,
               double v)
  {
	//return x / y / z / w / v / u / t / s;
  
	return pow(1. + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_5D {
  public:
    __device__ __host__ double
    operator()(double x,
               double y,
               double z,
               double w,
               double v)
    {
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return exp(
        -1.0 * (pow(25., 2.) * pow(x - beta, 2.) + 
				pow(25., 2.) * pow(y - beta, 2.) +
                pow(25., 2.) * pow(z - beta, 2.) + 
				pow(25., 2.) * pow(w - beta, 2.) +
                pow(25., 2.) * pow(v - beta, 2.)));
    }
};

class F_5_5D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m)
  {
	//return x / y / z / k / m / n / p / q;
  
	double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta);
    return exp(t1);
  }
};

class F_6_5D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y)
  {
	//return u / v / w / x / y / z / p / t;
	if (y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }
};

#endif
