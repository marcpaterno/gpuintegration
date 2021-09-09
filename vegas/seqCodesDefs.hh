#ifndef SEQ_CODES_DEFS_HH
#define SEQ_CODES_DEFS_HH

//#define ALPH 1.5
#define ALPH 0.5
#define NDMX 500
//#define NDMX 50
#define MXDIM 20
#define TINY 1.0e-30

extern long idum;
int iternumber = 0;

long idum = (-1);      /* for ranno */

int ndim;       /* for fxn */
float xoff;
#define NRANSI
#define NR_END 1
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define PI 3.14159265358979323846

static int imaxarg1, imaxarg2;
#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
		   (imaxarg1) : (imaxarg2))

static int iminarg1, iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
		   (iminarg1) : (iminarg2))

static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)


  bool
    adjustParams(unsigned long& ncall, int& totalIters)
  {
    if (ncall >= 6e9 && totalIters >= 100)
      return false;
    else if (ncall >= 6e9) {
      totalIters += 10;
      return true;
    } else {
      ncall += 1e9;
      return true;
    }
  }

#endif
