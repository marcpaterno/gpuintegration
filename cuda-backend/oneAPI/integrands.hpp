#ifndef SYCL_INTEGRANDS_HPP
#define SYCL_INTEGRANDS_HPP
#include <CL/sycl.hpp>


class Addition_8D {
  public:
    SYCL_EXTERNAL double
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
    SYCL_EXTERNAL double
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
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double w, double v, double m)
    {
	  return x + y + z + w + v + m;
    }
  };

class Addition_5D {
  public:
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double w, double v)
    {
	  return x + y + z + w + v;
    }
  };

class Addition_4D {
  public:
    SYCL_EXTERNAL double
    operator()(double x, double y, double z, double w)
    {
	  return x + y + z + w;
    }
  };

class Addition_3D {
  public:
    double
    operator()(double x, double y, double z)
    {
		return x + y + z;
    }
};

class SinSum_3D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z)
  {
    return sycl::sin(x + y + z);
  }
};

class SinSum_4D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k)
  {
    return sycl::sin(x + y + z + k);
  }
};

class SinSum_5D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k, double l)
  {
    return sycl::sin(x + y + z + k + l);
  }
};

class SinSum_6D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    return sycl::sin(x + y + z + k + l + m);
  }
};

class SinSum_7D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k, double l, double m, double n)
  {
    return sycl::sin(x + y + z + k + l + m + n);
  }
};

class SinSum_8D {
public:
  SYCL_EXTERNAL double
  operator()(double x, double y, double z, double k, double l, double m, double n, double p)
  {
    return sycl::sin(x + y + z + k + l + m + n + p);
  }
};

class F_1_8D {
public:
  SYCL_EXTERNAL double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
  {
    return sycl::cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y +
               8. * z);
  }
};

class F_2_8D {
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
	//return x / y / z / w / v / u / t / s;
  //return 1.;
    //return x + y + z + w + v + u + t + s;/*
	
	const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / (a*a)) + ((x - b)*(x - b)));
    const double term_2 = 1. / ((1. / (a*a)) + ((y - b)*(y - b)));
    const double term_3 = 1. / ((1. / (a*a)) + ((z - b)*(z - b)));
    const double term_4 = 1. / ((1. / (a*a)) + ((w - b)*(w - b)));
    const double term_5 = 1. / ((1. / (a*a)) + ((v - b)*(v - b)));
    const double term_6 = 1. / ((1. / (a*a)) + ((u - b)*(u - b)));
	const double term_7 = 1. / ((1. / (a*a)) + ((t - b)*(t - b)));
	const double term_8 = 1. / ((1. / (a*a)) + ((s - b)*(s - b)));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7 * term_8;
    return val;
	/*const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
    const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
    const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
    const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(w - b, 2.));
    const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(v - b, 2.));
    const double term_6 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(u - b, 2.));
	const double term_7 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(t - b, 2.));
	const double term_8 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(s - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7 * term_8;
    return val;*/
	
  }
};

class F_3_8D{
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
	//return x / y / z / w / v / u / t / s;
  
	return sycl::pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_8D {
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
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(u - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(t - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(s - beta, 2.)));
    }
};

class F_5_8D {
public:
  SYCL_EXTERNAL double
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
    double t1 = -10. * sycl::fabs(x - beta) - 10. * sycl::fabs(y - beta) -
                10. * sycl::fabs(z - beta) - 10. * sycl::fabs(k - beta) -
                10. * sycl::fabs(m - beta) - 10. * sycl::fabs(n - beta) -
                10. * sycl::fabs(p - beta) - 10. * sycl::fabs(q - beta);
    return sycl::exp(t1);
  }
};

class F_6_8D {
public:
  SYCL_EXTERNAL double
  operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	//return u / v / w / x / y / z / p / t;
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3 || t > .2)
      return 0.;
    else
      return sycl::exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. *p + 3. * t);
  }
};

class F_1_6D {
public:
  SYCL_EXTERNAL double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x)
  {
    return sycl::cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x);
  }
};

class F_2_6D {
public:
  SYCL_EXTERNAL double
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
    const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
    const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
    const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
    const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(w - b, 2.));
    const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(v - b, 2.));
    const double term_6 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(u - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

class F_3_6D{
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
  {
	//return x / y / z / w / v / u / t / s;
  
	return sycl::pow(1. + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_6D {
  public:
    SYCL_EXTERNAL double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
    {
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(u - beta, 2.)));
    }
};

class F_5_6D {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n)
  {
	//return x / y / z / k / m / n / p / q;
  
	double beta = .5;
    double t1 = -10. * sycl::fabs(x - beta) - 10. * sycl::fabs(y - beta) -
                10. * sycl::fabs(z - beta) - 10. * sycl::fabs(k - beta) -
                10. * sycl::fabs(m - beta) - 10. * sycl::fabs(n - beta);
    return sycl::exp(t1);
  }
};

class F_6_6D {
public:
  SYCL_EXTERNAL double
  operator()(double u, double v, double w, double x, double y, double z)
  {
	//return u / v / w / x / y / z / p / t;
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return sycl::exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }
};

class F_1_7D {
public:
  SYCL_EXTERNAL double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y)
  {
    return sycl::cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y);
  }
};

class F_2_7D {
public:
  SYCL_EXTERNAL double
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
    const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
    const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
    const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
    const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(w - b, 2.));
    const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(v - b, 2.));
    const double term_6 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(u - b, 2.));
	const double term_7 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(t - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7;
    return val;
  }
};

class F_3_7D{
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t)
  {
	//return x / y / z / w / v / u / t / s;
  
	return sycl::pow(1. + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_7D {
  public:
    SYCL_EXTERNAL double
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
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(u - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(t - beta, 2.)));
    }
};

class F_5_7D {
public:
  SYCL_EXTERNAL double
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
    double t1 = -10. * sycl::fabs(x - beta) - 10. * sycl::fabs(y - beta) -
                10. * sycl::fabs(z - beta) - 10. * sycl::fabs(k - beta) -
                10. * sycl::fabs(m - beta) - 10. * sycl::fabs(n - beta) -
                10. * sycl::fabs(p - beta);
    return sycl::exp(t1);
  }
};

class F_6_7D {
public:
  SYCL_EXTERNAL double
  operator()(double u, double v, double w, double x, double y, double z, double p)
  {
	//return u / v / w / x / y / z / p / t;
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3)
      return 0.;
    else
      return sycl::exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. *p);
  }
};

class F_1_5D {
public:
  SYCL_EXTERNAL double
  operator()(double s,
             double t,
             double u,
             double v,
             double w)
  {
    return sycl::cos(s + 2. * t + 3. * u + 4. * v + 5. * w);
  }
};

class F_2_5D {
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v)
  {
	//return x / y / z / w / v / u / t / s;
  
	const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
    const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
    const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
    const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(w - b, 2.));
    const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(v - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5;
    return val;
  }
};

class F_3_5D{
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v)
  {
	//return x / y / z / w / v / u / t / s;
  
	return sycl::pow(1. + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class F_4_5D {
  public:
    SYCL_EXTERNAL double
    operator()(double x,
               double y,
               double z,
               double w,
               double v)
    {
	  //return x / y / z / w / v / u / t / s;
	  double beta = .5;
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.)));
    }
};

class F_5_5D {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y,
             double z,
             double k,
             double m)
  {
	//return x / y / z / k / m / n / p / q;
  
	double beta = .5;
    double t1 = -10. * sycl::fabs(x - beta) - 10. * sycl::fabs(y - beta) -
                10. * sycl::fabs(z - beta) - 10. * sycl::fabs(k - beta) -
                10. * sycl::fabs(m - beta);
    return sycl::exp(t1);
  }
};

class F_6_5D {
public:
  SYCL_EXTERNAL double
  operator()(double u, double v, double w, double x, double y)
  {
	//return u / v / w / x / y / z / p / t;
	if (y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return sycl::exp(9. * y + 8. * x + 7. * w + 6. * v + 5. * u);
  }
};

class BoxIntegral8_15 {
public:
  SYCL_EXTERNAL double
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
    sum = sycl::pow(x, 2.) + sycl::pow(y, 2.) + sycl::pow(z, 2.) + sycl::pow(k, 2.) + sycl::pow(l, 2.) +
          sycl::pow(m, 2.) + sycl::pow(n, 2.) + sycl::pow(o, 2.);
    return sycl::pow(sum, s / 2.);
  }
};

class BoxIntegral8_22 {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    double s = 22.;
    double sum = 0;
    sum = sycl::pow(x, 2.) + sycl::pow(y, 2.) + sycl::pow(z, 2.) + sycl::pow(k, 2.) + sycl::pow(l, 2.) +
          sycl::pow(m, 2.) + sycl::pow(n, 2.) + sycl::pow(o, 2.);
    double exponent = s/2.;
	return sycl::powr(sum, exponent);
  }
};

class BoxIntegral8_25 {
public:
  SYCL_EXTERNAL double
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
    sum = sycl::pow(x, 2.) + sycl::pow(y, 2.) + sycl::pow(z, 2.) + sycl::pow(k, 2.) + sycl::pow(l, 2.) +
          sycl::pow(m, 2.) + sycl::pow(n, 2.) + sycl::pow(o, 2.);
    return sycl::pow(sum, s / 2.);
  }
};

#endif
