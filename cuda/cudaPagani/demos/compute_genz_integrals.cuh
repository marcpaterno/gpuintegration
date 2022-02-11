
#include <math.h>       /* atan */

double r8_abs ( double x )

//****************************************************************************80
//
//  Purpose:
//
//    R8_ABS returns the absolute value of an R8.
//
//  Modified:
//
//    14 November 2006
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, the quantity whose absolute value is desired.
//
//    Output, double R8_ABS, the absolute value of X.
//
{
  double value;

  if ( 0.0 <= x )
  {
    value = x;
  } 
  else
  {
    value = -x;
  }
  return value;
}

//****************************************************************************80
//
//  Purpose:
//
//    GENZ_PHI estimates the normal cumulative density function.
//
//  Discussion:
//
//    The approximation is accurate to 1.0E-07.
//
//    This routine is based upon algorithm 5666 for the error function,
//    from Hart et al.
//
//  Modified:
//
//    20 March 2007
//
//  Author:
//
//    Original FORTRAN77 version by Alan Miller
//    C++ version by John Burkardt
//
//  Reference:
//
//    John Hart, Ward Cheney, Charles Lawson, Hans Maehly,
//    Charles Mesztenyi, John Rice, Henry Thatcher,
//    Christoph Witzgall,
//    Computer Approximations,
//    Wiley, 1968,
//    LC: QA297.C64.
//
//  Parameters:
//
//    Input, double Z, a value which can be regarded as the distance,
//    in standard deviations, from the mean.
//
//    Output, double GENZ_PHI, the integral of the normal PDF from negative
//    infinity to Z.
//
//  Local parameters:
//
//    Local, double ROOTPI, despite the name, is actually the 
//    square root of TWO * pi.
//
double genz_phi ( double z )
{
  double expntl;
  double p;
  const double p0 = 220.2068679123761;
  const double p1 = 221.2135961699311;
  const double p2 = 112.0792914978709;
  const double p3 = 33.91286607838300;
  const double p4 = 6.373962203531650;
  const double p5 = 0.7003830644436881;
  const double p6 = 0.03526249659989109;
  const double q0 = 440.4137358247522;
  const double q1 = 793.8265125199484;
  const double q2 = 637.3336333788311;
  const double q3 = 296.5642487796737;
  const double q4 = 86.78073220294608;
  const double q5 = 16.06417757920695;
  const double q6 = 1.755667163182642;
  const double q7 = 0.08838834764831844;
  const double rootpi = 2.506628274631001;
  double zabs;

  zabs = r8_abs ( z );
//
//  12 < |Z|.
//
  if ( 12.0 < zabs )
  {
    p = 0.0;
  }
  else
  {
//
//  |Z| <= 12
//
    expntl = exp ( - zabs * zabs / 2.0 );
//
//  |Z| < 7
//
    if ( zabs < 7.0 )
    {
      p = expntl * (((((( 
                  p6 
         * zabs + p5 ) 
         * zabs + p4 ) 
         * zabs + p3 ) 
         * zabs + p2 ) 
         * zabs + p1 ) 
         * zabs + p0 ) / ((((((( 
                  q7 
         * zabs + q6 ) 
         * zabs + q5 ) 
         * zabs + q4 ) 
         * zabs + q3 ) 
         * zabs + q2 ) 
         * zabs + q1 ) 
         * zabs + q0 );
    }
//
//  CUTOFF <= |Z|
//
    else
    {
      p = expntl / ( 
        zabs + 1.0 / (
        zabs + 2.0 / ( 
        zabs + 3.0 / ( 
        zabs + 4.0 / ( 
        zabs + 0.65 ))))) / rootpi;
    }
  }

  if ( 0.0 < z )
  {
    p = 1.0 - p;
  }

  return p;
}


template<size_t ndim>
double compute_product_peak(std::array<double, ndim> alphas, std::array<double, ndim> betas){
    double value = 1.0;

    for (size_t j = 0; j < ndim; j++ )
    {
      value = value * alphas[j] * ( 
          atan ( ( 1.0 - betas[j] ) * alphas[j] ) 
        + atan (       + betas[j]   * alphas[j] ) );
    }
    
    return value;
}

template<size_t ndim>
double compute_gaussian(std::array<double, ndim> alphas, std::array<double, ndim> betas)
{
    
    double value = 1.0;
    const double pi = 3.14159265358979323844;
    double ab = sqrt ( 2.0 );
    for (size_t j = 0; j < ndim; j++ )
    {
      value = value * ( sqrt ( pi ) / alphas[j] ) * 
        (   genz_phi ( ( 1.0 - betas[j] ) * ab * alphas[j] ) 
          - genz_phi (       - betas[j]   * ab * alphas[j] ) );
    }
    return value;
}

template<size_t ndim>
double compute_c_zero(std::array<double, ndim> alphas, std::array<double, ndim> betas)
{
    double value = 1.0;
    for (size_t j = 0; j < ndim; j++ )
    {
      double ab = alphas[j] * betas[j];
      value = value * 
        ( 2.0 - exp ( - ab ) - exp ( ab - alphas[j] ) ) / alphas[j];
    }
    return value;
}

template<size_t ndim>
double
compute_discontinuous(std::array<double, ndim> alphas, std::array<double, ndim> betas){
  
    double value = 1.0;
    for (size_t j = 0; j < ndim; j++ )
    {
      value = value * ( exp ( alphas[j] * betas[j] ) - 1.0 ) / alphas[j];
    }
    return value;
}

