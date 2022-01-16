#ifndef MCUBES_SEQ_HH
#define MCUBES_SEQ_HH

/* Psuedo code 
for each iteration:
for each subcube:
for each ndim:
calculate the probability of that subcube using random number
calculate the function value in each subcube using the probability of each dim
filling out the d array
adjusting the intervals
*/

/* Driver for routine vegas, shorter version
   to avoid differnt cases */
   
#include "vegas/seqCodesDefs.hh"
#include "cudaPagani/quad/util/cuhreResult.cuh"
#include "cudaPagani/quad/util/Volume.cuh"

#include <stdio.h>

#include <stdlib.h>
#include <math.h>
#define NRANSI
#include <fstream>
#include <string>
#include <iostream>
#include <tuple>
#define OUTPUT 3

 
/* receives the array and the size and returns the revers of the array for type string
   used for reading and writing files*/
void reverse_bytes(void *data, size_t size)
{
  char *i, *j;
  char tmp;
  for (i = (char*)data, j = i + size - 1; i < j; i++, j--) {
    tmp = *i;
    *i = *j;
    *j = tmp;
  }
}
/* receives the array and the size and returns the revers of the array for type int
   not used in this code*/
void revArrInt(int32_t* arr, size_t start, size_t end)
{
  int32_t temp;
  while (start < end)
    {
      temp = arr[start];
      arr[start] = arr[end];
      arr[end] = temp;
      start++;
      end--;
    }
}
/* receives the array and the size and returns the revers of the array for type double
   not used in this code*/
void revArrDouble(double* arr, size_t start, size_t end)
{
  double temp;
  while (start < end)
    {
      temp = arr[start];
      arr[start] = arr[end];
      arr[end] = temp;
      start++;
      end--;
    }
}
//not used in this code
void extract(FILE* fp, void *data, size_t size)
{
  fread(data, size, 1, fp);
  reverse_bytes(data, size);
}
//not used in this code
void writebin(FILE* fp, void *data, size_t size)
{
  reverse_bytes(data, size);
  fwrite(data, size, 1, fp);
}


//function to create a vector
double *vector(long nl, long nh)
{
  double *v;

  v = (double *)malloc((size_t) ((nh - nl + 1 + NR_END) * sizeof(double)));
  //if (!v) nrerror("allocation failure in vector()");
  return v - nl + NR_END;
}


//generates random number, receives a pointer, returns a random number
double ran2(long *idum)
{
  int j;
  long k;
  static long idum2 = 123456789;
  static long iy = 0;
  static long iv[NTAB];
  double temp;

  if (*idum <= 0) {
    if (-(*idum) < 1) *idum = 1;
    else *idum = -(*idum);
    idum2 = (*idum);
    for (j = NTAB + 7; j >= 0; j--) {
      k = (*idum) / IQ1;
      *idum = IA1 * (*idum - k * IQ1) - k * IR1;
      if (*idum < 0) *idum += IM1;
      if (j < NTAB) iv[j] = *idum;
    }
    iy = iv[0];
  }
  k = (*idum) / IQ1;
  *idum = IA1 * (*idum - k * IQ1) - k * IR1;
  if (*idum < 0) *idum += IM1;
  k = idum2 / IQ2;
  idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
  if (idum2 < 0) idum2 += IM2;
  j = iy / NDIV;
  iy = iv[j] - idum2;
  iv[j] = *idum;
  if (iy < 1) iy += IMM1;
  if ((temp = AM * iy) > RNMX) return RNMX;
  else return temp;
}
/*
double fxn(double *rn, double wgt) {
int j;
double tsum = 0.0;
for (j = 0; j < ndim; j++) {
tsum += rn[j];
}
return sin(tsum);

}
*/



// the function that we want to calculate the multidimensional integral for. receives am array of random numbers, each element is for each dimension
/*double fxn(double x[])
{
  iternumber += 1;
  double a = 0.1;
  double k;
  int j;
  k = (a * sqrt(M_PI));
  k = 1.0 / k;
  k = pow(k, 4);
  double tsum = 0.0;
  for (j = 1; j < 5; j++) {
    tsum += (x[j] - 0.5) * (x[j] - 0.5) / (a * a);
  }
  tsum = exp(-tsum);
  //printf("iter: %d , function value: %.12f\n ", iternumber, x[0]);
  return (tsum * k);
}*/


/*
double fxn(double pt[],double wgt)
{
int j;
double ans,sum;

for (sum=0.0,j=1;j<=ndim;j++) sum += (100.0*SQR(pt[j]-xoff));
ans=(sum < 80.0 ? exp(-sum) : 0.0);
ans *= pow(5.64189,(double)ndim);
return ans;
}
*/



void rebin(double rc, int nd, double r[], double xin[], double xi[])

{

  int i, k = 0;

  double dr = 0.0, xn = 0.0, xo = 0.0;



  for (i = 1; i < nd; i++) {

    while (rc > dr)

      dr += r[++k];

    if (k > 1) xo = xi[k - 1];

    xn = xi[k];

    dr -= rc;

    xin[i] = xn - (xn - xo) * dr / r[k];

  }

  for (i = 1; i < nd; i++) xi[i] = xin[i];

  xi[nd] = 1.0;

  // for (i=1;i<=nd;i++) printf("bins edges: %.10f\n", xi[i]);
  // printf("---------------------\n");
}


/*receives each subcube, the indexing array, the number of ng(ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim)) and the number of intervals in each dimension
  calculates the indexes for that subcube*/
void get_indx(int ms, int *da, int ND, int NINTV) {
  int dp[MXDIM];
  int j, t0, t1;
  int m = ms;
  dp[0] = 1;
  dp[1] = NINTV;
  // avoid use of pow() function
  for (j = 0; j < ND - 2; j++) {
    dp[j + 2] = dp[j + 1] * NINTV;
  }
  //
  for (j = 0; j < ND; j++) {
    t0 = dp[ND - j - 1];
    t1 = m / t0;
    da[j] = 1 + t1;

    m = m - t1 * t0;
  }

}

/*receives the regn array, number of dimension of the function, the pointer to the function, ncall variable which is 1000 to calculate ng and npg, 
a pointer variable tgral that holds the integral calculated, si and chi2a for calculating the variance
This function calculates the integral and variance and adjust the bins in each iteration.  
*/


template<typename IntegT, int NDIM>
void vegas_mcubes(IntegT integrand, double regn[], int ndim, int init,
           unsigned long ncall, int itmx, int nprn, double *tgral, double *sd,
           double *chi2a)
{
  /*std::ofstream xi_outfile;
  xi_outfile.open("mseq_xi.csv");
  xi_outfile <<"iter, dim, binID, left, right, contribution\n"; 
      
  std::ofstream eval_outfile;
  eval_outfile.open("mseq_eval.csv");
  
  eval_outfile <<"iter, cube, npg, dim1, dim2, dim3, dim4, dim5, dim6, tmp, f\n";
  
  std::ofstream intervals_outfile;
  intervals_outfile.open("mcubes_intervals.csv");
  intervals_outfile<<"iter, kg1, kg2, kg3, kg4, kg5, kg6\n";*/
  std::cout<<"ncall:"<<ncall<<"\n";
  static int i, it, j, k, nd, ndo, ng, npg, ia[MXDIM + 1], kg[MXDIM + 1];
  static double calls, dv2g, dxg, f, f2, f2b, fb, rc, ti, tsi, wgt, xjac, xn, xnd, xo;
  static double d[NDMX + 1][MXDIM + 1], di[NDMX + 1][MXDIM + 1], dt[MXDIM + 1],
    dx[MXDIM + 1], r[NDMX + 1], x[MXDIM + 1], xi[MXDIM + 1][NDMX + 1], xin[NDMX + 1];
  static double schi, si, swgt;

  // code works only  for (2 * ng - NDMX) >= 0)
  // for (int i = 0; i < 20; i++) {
  // printf("ridx,rn:    %d   %.15f\n", i, randa[i]);
  // }
  ndo = 1;
  for (j = 1; j <= ndim; j++) 
	  xi[j][1] = 1.0;

  si = swgt = schi = 0.0;
  nd = NDMX;
  ng = 1;

  ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim);

  for (k = 1, i = 1; i <= ndim; i++) 
	  k *= ng;
  npg = IMAX(ncall / k, 2);
  calls = (double)npg * (double)k;
  dxg = 1.0 / ng;
  
  for (dv2g = 1, i = 1; i <= ndim; i++) dv2g *= dxg;
  dv2g = SQR(calls * dv2g) / npg / npg / (npg - 1.0);
  xnd = nd;
  dxg *= xnd;
  //std::cout<<"dxg*xnd:"<<dxg<<"\n";
  xjac = 1.0 / calls;
  for (j = 1; j <= ndim; j++) {
    dx[j] = regn[j + ndim] - regn[j];
    //printf("setting dx (dim range) %e, %e\n", dx[j], xjac);
    xjac *= dx[j];
  }

  for (i = 1; i <= IMAX(nd, ndo); i++) 
	  r[i] = 1.0;
  for (j = 1; j <= ndim; j++) 
	  rebin(ndo / xnd, nd, r, xin, xi[j]);
  ndo = nd;
  

  //printf("ng, npg, dv2g, xjac, %d, %d, %e, %e\n", ng, npg, dv2g, xjac);
  
  double ran00;
  int ncubes = pow(ng, ndim);
  
  /*std::cout<<"ndim:"<<ndim<<"\n";
  std::cout<<"calls:"<<calls<<"\n";
  std::cout<<"ncall:"<<ncall<<"\n";
  std::cout<<"MXDIM:"<<MXDIM<<"\n";
  std::cout<<"NDMX:"<<NDMX<<"\n";
  std::cout<<"ng:"<<ng<<"\n";
  std::cout<<"npg:"<<npg<<"\n";
  std::cout<<"ncubes:"<<ncubes<<"\n";
  std::cout<<"dxg:"<<dxg<<"\n";
  std::cout<<"xjac:"<<xjac<<"\n";*/
  
  for (it = 1; it <= itmx; it++) {
    ti = tsi = 0.0; //variables for aggregation sum of f and f^2 in all the subcubes
    
    //std::cout<<"starting to reset contributions for new iteration\n";
    for (j = 1; j <= ndim; j++) {
      for (i = 1; i <= nd; i++) 
		  d[i][j] = di[i][j] = 0.0;
    }
    
    //std::cout<<"finished resetting contributions for new iteration\n";
    
    for (int m = 0; m < ncubes; m++) {
      get_indx(m, &kg[1], ndim, ng); //replaced the line above with this, edited by Ioannis  
      fb = f2b = 0.0;
      
	  for (k = 1; k <= npg; k++) {
        wgt = xjac;
        for (j = 1; j <= ndim; j++) {
            
            ran00 = ran2(&idum);
            xn = (kg[j] - ran00) * dxg + 1.0;

            ia[j] = IMAX(IMIN((int)(xn), NDMX), 1);
	  
            //getting the lb and rb for each bin for calculating the probability
            if (ia[j] > 1) {
                xo=xi[j][ia[j]]-xi[j][ia[j]-1];  //calculating the length of the bin
                rc=xi[j][ia[j]-1]+(xn-ia[j])*xo;
            } else {
                xo=xi[j][ia[j]];
                rc=(xn-ia[j])*xo;   
            }

            x[j] = regn[j] + rc * dx[j]; //generating random number for every dimension
            wgt *= xo * xnd;  // calculating probability xnd=ndmx = 50
        }
  
        std::array<double, NDIM> xx;
        for(int dim = 0; dim<NDIM; ++dim)
          xx[dim]=x[dim+1];
        double tmp = std::apply(integrand, xx);

        f = wgt * tmp;
  
        f2 = f * f;
        fb += f;
        f2b += f2;
        for (j = 1; j <= ndim; j++) {
          di[ia[j]][j] += f;
          d[ia[j]][j] += f2;
        }
        
        //std::cout<<"computed contributions\n";
      }
      
      f2b = sqrt(f2b * npg);
      f2b = (f2b - fb) * (f2b + fb);
      if (f2b <= 0.0) f2b = TINY;
      ti += fb;
      tsi += f2b;

    } // end of subcube loop
    //calculating the integral in this part
    //"iter, dim, dim_binID, global_binID, left, right, contribution, damped_contr\n";
      //xi[MXDIM + 1][NDMX + 1]
	/*if(OUTPUT <= 3){
		for(int dim=1; dim< MXDIM+1; dim++)
			for(int bin=1; bin<NDMX+1; bin++){
			xi_outfile << it << ", " << dim << "," << bin << ","  
				<< -1 << xi[dim][bin-1] << "," << xi[dim][bin] << ","
				<< d[bin][dim] << "," << "-1\n";
			}
	}*/
    
    tsi *= dv2g;
    wgt = 1.0 / tsi;
    si += wgt * ti;
    schi += wgt * ti * ti;
    swgt += wgt;
    *tgral = si / swgt;
    *chi2a = (schi - si * (*tgral)) / (it - 0.9999);
    
    if (*chi2a < 0.0) 
        *chi2a = 0.0;
    
    *sd = sqrt(1.0 / swgt);
    tsi = sqrt(tsi);

    /* printf("%s %3d : integral = %14.7g +/-  %9.2g\n",
            " iteration no.", it, ti, tsi);
     printf("%s integral =%14.7g+/-%9.2g chi**2/IT n = %9.2g\n",
            " all iterations:  ", *tgral, *sd, *chi2a);*/

    //adjusting the intervals 
    for (j = 1; j <= ndim; j++) {
      xo = d[1][j];
      xn = d[2][j];
      d[1][j] = (xo + xn) / 2.0;
      dt[j] = d[1][j];
      for (i = 2; i < nd; i++) {
        rc = xo + xn;
        xo = xn;
        xn = d[i + 1][j];
        d[i][j] = (rc + xn) / 3.0;
        dt[j] += d[i][j];
      }
      d[nd][j] = (xo + xn) / 2.0;
      dt[j] += d[nd][j];
    }
    
    for (j = 1; j <= ndim; j++) {
      rc = 0.0;
      for (i = 1; i <= nd; i++) {
        if (d[i][j] < TINY) d[i][j] = TINY;
            r[i] = pow((1.0 - d[i][j] / dt[j]) /
		   (log(dt[j]) - log(d[i][j])), ALPH);
            rc += r[i];
      }
      rebin(rc / xnd, nd, r, xin, xi[j]);

    }   
  }
    //xi_outfile.close();
    //eval_outfile.close();
}

template<typename IntegT, int NDIM>
cuhreResult<double> seq_mcubes_integrate(IntegT integrand, int ndim, double epsrel, double epsabs, unsigned long ncall, quad::Volume<double, NDIM> const* volume, int itmx)
{
  //declaring the variables
  cuhreResult<double> result;
  int init, j/*, ncall*/, nprn;
  double avgi, chi2a, sd;
  double *regn;
  init = -1;
  
  regn = vector(1, 20);
  //printf("IDUM=\n");
  //scanf("%ld",&idum);
  // if (idum > 0) idum = -idum;
  // idum = 1.0;
  // if (idum > 0) idum = -idum;

  // printf("ENTER NDIM,XOFF,NCALL,ITMAX,NPRN\n");
  // scanf("%d %f %d %d %d",&ndim,&xoff,&ncall,&itmax,&nprn);
  
  //initializing the variables and the arrays
  ndim = NDIM;
  //ncall = 1000;
  //itmax = 10;
  nprn = -1;
  avgi = sd = chi2a = 0.0;
  
  for (j = 1; j <= ndim; j++) {
    regn[j] = volume->lows[j-1];
    regn[j + ndim] = volume->highs[j-1];
    //printf("vol[%i]:(%f,%f)\n", j, regn[j], regn[j + ndim]);
    
  }

  init = -1;
  //calling the vegas function to calculate the integral and adjusting the intervals
  vegas_mcubes<IntegT, NDIM>(integrand, regn, NDIM, init, ncall, itmx, nprn, &avgi, &sd, &chi2a);
  //printf("Number of iterations performed: %d\n", itmx);
  //printf("Integral, Standard Dev., Chi-sq. = %12.6f %12.6f% 12.6f\n", avgi, sd, chi2a);
  // init = 1;
  // vegas(regn,ndim,fxn,init,ncall,itmax,nprn,&avgi,&sd,&chi2a);
  // printf("Additional iterations performed: %d \n",itmx);
  // printf("Integral, Standard Dev., Chi-sq. = %12.6f %12.6f% 12.6f\n",
  // avgi,sd,chi2a);

  //free_vector(regn,1,20);
  //printf("Normal completion\n");
  result.estimate = static_cast<double>(avgi);
  result.errorest = static_cast<double>(sd);
  result.chi_sq = static_cast<double>(chi2a);
  result.status = sd/abs(avgi) <= epsrel || sd <= epsabs ? 0 : 1;
  return result;
}
#undef NRANSI

#endif
