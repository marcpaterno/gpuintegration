
/* Driver for routine vegas */

#include <stdio.h>
//#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#define NRANSI
//#include "nr.h"
//#include "nrutil.h"
#include <tuple>

#define ALPH 1.5
//#define NDMX 50
#define NDMX 500
#define MXDIM 10
#define TINY 1.0e-30
//#include "cudaPagani/quad/util/cudaMemoryUtil.h"

extern long idum;
int iternumber = 0;

long idum = (-1);      /* for ranno */

int ndim;       /* for fxn */
float xoff;


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


static int imaxarg1,imaxarg2;
#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
		   (imaxarg1) : (imaxarg2))

static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
		   (iminarg1) : (iminarg2))

static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)



float *vector(long nl, long nh)
{
  float *v;

  v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
  //if (!v) nrerror("allocation failure in vector()");
  return v-nl+NR_END;
}



float ran2(long *idum)
{
  int j;
  long k;
  static long idum2=123456789;
  static long iy=0;
  static long iv[NTAB];
  float temp;

  if (*idum <= 0) {
    if (-(*idum) < 1) *idum=1;
    else *idum = -(*idum);
    idum2=(*idum);
    for (j=NTAB+7;j>=0;j--) {
      k=(*idum)/IQ1;
      *idum=IA1*(*idum-k*IQ1)-k*IR1;
      if (*idum < 0) *idum += IM1;
      if (j < NTAB) iv[j] = *idum;
    }
    iy=iv[0];
  }
  k=(*idum)/IQ1;
  *idum=IA1*(*idum-k*IQ1)-k*IR1;
  if (*idum < 0) *idum += IM1;
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2;
  if (idum2 < 0) idum2 += IM2;
  j=iy/NDIV;
  iy=iv[j]-idum2;
  iv[j] = *idum;
  if (iy < 1) iy += IMM1;
  if ((temp=AM*iy) > RNMX) return RNMX;
  else return temp;
}
/*
float fxn(float *rn, float wgt) {
int j;
float tsum = 0.0;
for (j = 0; j < ndim; j++) {
tsum += rn[j];
}
return sin(tsum);

}
*/




float fxn(float x[], float wgt)
{
  iternumber += 1;
  float a = 0.1;
  float k;
  int j;
  k = (a * sqrt(M_PI));
  k = 1.0 / k;
  k = pow(k, 4);
  float tsum = 0.0;
  for (j = 1; j < 5; j++) {
    tsum += (x[j] - 0.5) * (x[j] - 0.5) / (a * a);
  }
  tsum = exp(-tsum);
  //printf("iter: %d , function value: %.12f\n ", iternumber, x[0]);
  return (tsum * k);
  

}
/*
float fxn(float pt[],float wgt)
{
int j;
float ans,sum;

for (sum=0.0,j=1;j<=ndim;j++) sum += (100.0*SQR(pt[j]-xoff));
ans=(sum < 80.0 ? exp(-sum) : 0.0);
ans *= pow(5.64189,(double)ndim);
return ans;
}
*/



void rebin(float rc, int nd, float r[], float xin[], float xi[])

{

  int i,k=0;

  float dr=0.0,xn=0.0,xo=0.0;

 

  for (i=1;i<nd;i++) {

    while (rc > dr)

      dr += r[++k];

    if (k > 1) xo=xi[k-1];

    xn=xi[k];

    dr -= rc;

    xin[i]=xn-(xn-xo)*dr/r[k];

  }

  for (i=1;i<nd;i++) xi[i]=xin[i];

  xi[nd]=1.0;
}

template<typename IntegT, int NDIM>
  void vegas(IntegT integrand, float regn[], int ndim, int init,
	   unsigned long ncall, int itmx, int nprn, float *tgral, float *sd,
	   float *chi2a)
{
  
  
  static int i,it,j,k,mds,nd,ndo,ng,npg,ia[MXDIM+1],kg[MXDIM+1];
  static float calls,dv2g,dxg,f,f2,f2b,fb,rc,ti,tsi,wgt,xjac,xn,xnd,xo;
  static float d[NDMX+1][MXDIM+1],di[NDMX+1][MXDIM+1],dt[MXDIM+1],
    dx[MXDIM+1], r[NDMX+1],x[MXDIM+1],xi[MXDIM+1][NDMX+1],xin[NDMX+1];
  static double schi,si,swgt;
  float ran00 ; 
  if (init <= 0) {
    mds=ndo=1;
    for (j=1;j<=ndim;j++) xi[j][1]=1.0;
  }
  if (init <= 1) 
    si=swgt=schi=0.0;
  
  if (init <= 2) {
    nd=NDMX;
    ng=1;
    if (mds) {
      ng=(int)pow(ncall/2.0+0.25,1.0/ndim);
      mds=1;
      if ((2*ng-NDMX) >= 0) {
	mds = -1;
	npg=ng/NDMX+1;
	nd=ng/npg;
	ng=npg*nd;
      }
    }

    for (k=1,i=1;i<=ndim;i++) 
      k *= ng;
    npg=IMAX(ncall/k,2);
    calls=(float)npg * (float)k;
    dxg=1.0/ng;
    
    for (dv2g=1,i=1;i<=ndim;i++) 
      dv2g *= dxg;
    
    dv2g=SQR(calls*dv2g)/npg/npg/(npg-1.0);
    xnd=nd;
    dxg *= xnd;
    xjac=1.0/calls;
    
    for (j=1;j<=ndim;j++) {
      dx[j]=regn[j+ndim]-regn[j];
      xjac *= dx[j];
    }
    
    if (nd != ndo) {
      for (i=1;i<=IMAX(nd,ndo);i++) r[i]=1.0;
      for (j=1;j<=ndim;j++) rebin(ndo/xnd,nd,r,xin,xi[j]);
      ndo=nd;
    }
    
    if (nprn >= 0) {
      printf("%s:  ndim= %3d  ncall= %8.0f\n",
	     " Input parameters for vegas",ndim,calls);
      printf("%28s  it=%5d  itmx=%5d\n"," ",it,itmx);
      printf("%28s  nprn=%3d  ALPH=%5.2f\n"," ",nprn,ALPH);
      printf("%28s  mds=%3d  nd=%4d\n"," ",mds,nd);
      for (j=1;j<=ndim;j++) {
	printf("%30s xl[%2d]= %11.4g xu[%2d]= %11.4g\n",
	       " ",j,regn[j],j,regn[j+ndim]);
      }
    }
  }
  
  std::array<double, NDIM> xx;

  for (it=1;it<=itmx;it++) {
    
    ti=tsi=0.0;
    
    for (j=1;j<=ndim;j++) {
      kg[j]=1;
      for (i=1;i<=nd;i++) 
	d[i][j]=di[i][j]=0.0;
    }
    
    int counter = 0;
    for (;;) {
      
      //printf("entered main for loop\n");
      fb=f2b=0.0;
      for (k=1;k<=npg;k++) {
	//printf("entered npg for loop\n");
	wgt=xjac;
	for (j=1;j<=ndim;j++) {
	  //ran00 = (float)rand()/RAND_MAX;

	  xn=(kg[j]-ran2(&idum))*dxg+1.0;
	  
	  
	  ia[j]=IMAX(IMIN((int)(xn),NDMX),1);
	  if (ia[j] > 1) {
	    xo=xi[j][ia[j]]-xi[j][ia[j]-1];
	    rc=xi[j][ia[j]-1]+(xn-ia[j])*xo;
	  } else {
	    xo=xi[j][ia[j]];
	    rc=(xn-ia[j])*xo;
	  }
	  x[j]=regn[j]+rc*dx[j];
	  wgt *= xo*xnd;
	}
	
        for(int dim=0; dim<NDIM; ++dim){
          //printf("xx[%i]:%f\n", dim, x[dim+1]);
	  xx[dim]=x[dim+1];
	}
	
	f = std::apply(integrand, xx);
	//f=wgt*(*fxn)(x,wgt);
	f2=f*f;
	fb += f;
	f2b += f2;
	for (j=1;j<=ndim;j++) {
	  di[ia[j]][j] += f;
	  if (mds >= 0) d[ia[j]][j] += f2;
	}
	//printf("processed dim %i\n", j);
      }

      f2b=sqrt(f2b*npg);
      f2b=(f2b-fb)*(f2b+fb);
      
      if (f2b <= 0.0) 
	f2b=TINY;
      
      ti += fb;
      tsi += f2b;
      counter++;
      if (mds < 0) {
	for (j=1;j<=ndim;j++) d[ia[j]][j] += f2b;
      }
      
      for (k=ndim;k>=1;k--) {
	kg[k] %= ng;
	if (++kg[k] != 1) break;
      }
      if (k < 1) break;
    }
    
    tsi *= dv2g;
    wgt=1.0/tsi;
    si += wgt*ti;
    schi += wgt*ti*ti;
    swgt += wgt;
    *tgral=si/swgt;
    *chi2a=(schi-si*(*tgral))/(it-0.9999);
    if (*chi2a < 0.0) *chi2a = 0.0;
    *sd=sqrt(1.0/swgt);
    tsi=sqrt(tsi);
    if (nprn >= 0) {
      printf("%s %3d : integral = %14.7g +/-  %9.2g\n",
	     " iteration no.",it,ti,tsi);
      printf("%s integral =%14.7g+/-%9.2g chi**2/IT n = %9.2g\n"," all iterations:  ",*tgral,*sd,*chi2a);
      if (nprn) {
	for (j=1;j<=ndim;j++) {
	  printf(" DATA FOR axis  %2d\n",j);
	  printf("%6s%13s%11s%13s%11s%13s\n","X","delta i","X","delta i","X","delta i");
	  for (i=1+nprn/2;i<=nd;i += nprn+2) {
	    printf("%8.5f%12.4g%12.5f%12.4g%12.5f%12.4g\n",
		   xi[j][i],di[i][j],xi[j][i+1],
		   di[i+1][j],xi[j][i+2],di[i+2][j]);
	  }
	}
      }
    }

    for (j=1;j<=ndim;j++) {
      xo=d[1][j];
      xn=d[2][j];
      d[1][j]=(xo+xn)/2.0;
      dt[j]=d[1][j];
      for (i=2;i<nd;i++) {
	rc=xo+xn;
	xo=xn;
	xn=d[i+1][j];
	d[i][j] = (rc+xn)/3.0;
	dt[j] += d[i][j];
      }
      d[nd][j]=(xo+xn)/2.0;
      dt[j] += d[nd][j];
    }

    for (j=1;j<=ndim;j++) {
      rc=0.0;
      for (i=1;i<=nd;i++) {
	if (d[i][j] < TINY) d[i][j]=TINY;
	r[i]=pow((1.0-d[i][j]/dt[j])/
		 (log(dt[j])-log(d[i][j])),ALPH);
	rc += r[i];
      }
      rebin(rc/xnd,nd,r,xin,xi[j]);
    }
  }
}



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



template<typename IntegT, int NDIM>
  cuhreResult<double> vegas_book_integrate(IntegT integrand, 
					   int ndim, double epsrel, double epsabs, unsigned long ncall, 
					   quad::Volume<double, NDIM> const* volume, int itmx){
  printf("within vegas_book_integrate\n");
  int init = -1; //used by vegas function
  cuhreResult<double> result;
  int nprn = 0;
  bool status = 1;
  float regn[NDIM*2+2];

  for(int dim =1; dim<=NDIM; dim++){
    regn[dim]=volume->lows[dim-1];
    regn[dim+ndim]=volume->highs[dim-1];
    printf("setting bounds dim %i %f,%f (%i,%i)\n", dim,volume->lows[dim-1],volume->highs[dim-1], dim, dim+ndim);  
  }

  float estimate;
  float errorest;
  float chi_sq;
  //IntegT* d_integrand = quad::cuda_copy_to_managed(integrand);

  do {

    estimate = 0.;
    errorest = 0.;
    chi_sq = 0.;
    vegas<IntegT, NDIM>(integrand,
                        regn,
                        NDIM,
                        init,
                        ncall,
                        itmx,
                        nprn,
                        &estimate,
                        &errorest,
                        &chi_sq);
    status = abs(result.errorest/result.estimate) <= epsrel || result.errorest <= epsabs;
  } while (status == 1 && adjustParams(ncall, itmx) == true);
  
  result.estimate = static_cast<double>(estimate);
  result.errorest = static_cast<double>(errorest);
  result.chi_sq = static_cast<double>(chi_sq);
  return result;
}


#undef NRANSI
