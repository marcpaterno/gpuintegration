// this version handles multiple iterations where each iteration handles ndim dimensions
// It prints out a weighted integral estimate, variance, std.dev and chi2/dof after each dim
// of each iteration
// For this purpose we maintain three running sums over iterations
// (i) wsInt = running weighted sum of individual integral estimates
// (ii) wsIntSq = running weighted sum of squares of individual integral estimates
// (iii) Swgt = running sum of 1/Var[l] (also called weights)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <iostream>
#include <chrono>	
#include "cudaCuhre/demos/function.cuh"
/*#define MXDIM 20
#define ALPH 1.5
//#define ITMAX 32
//#define ITGRIDADJ 25
#define ITMAX 10
#define ITGRIDADJ 0
#define TINY 1.0e-30
#define N 4096       // N is number of slices
#define ndim 8             // ndim is number of dimensions
#define s 4                 // s for BNS(n,s) function
#define RANGESIN 10.0        //upper limit for sin integral*/

/*double f4(double x[]);
double f5(double x[]);
double funcgauss(double x[]);
double newfuncgauss(double x[]);
double funcbns(double y[]);
double funcsin(double x[]);
double funcrangesin(double x[]);
void printgrid(double y[ndim + 1][N + 1]);
void rebin(double rc, int nd, double r[], double xin[], double xi[]);*/

void rebin(double rc, int nd, double r[], double xin[], double xi[]) {
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
}

double f4(double x[]) {
    int i; double value;
    value = 0.0;
	int ndim = 8;
    for (i = 1; i <= ndim; i++) {
        value += -(pow(25.0, 2.0) * pow((x[i] - 0.5), 2.0));
    }
    value = exp(value);
    return (value);
}

double f5(double x[]) {
    int i; double value;
	int ndim = 8;
    value = 0.0;
    for (i = 1; i <= ndim; i++) {
        value += (-10.0 * abs(x[i] - 0.5));
    }
    value = exp(value);
    return (value);
}

double funcgauss(double x[]) {
    double sigma = 0.03;
    double mu = 0.5;
    double tsum = 0.0; double k;
    int j;
	int ndim = 8;
    k = sigma * sqrt(2.0 * M_PI);
    k = pow(k, ndim);
    k = 1.0 / k;
    for (j = 1; j <= ndim; j++) {
        tsum += (x[j] - mu) * (x[j] - mu);
    }
    tsum = tsum / (2 * sigma * sigma);
    tsum = exp(-tsum);
    return (tsum * k);
}

double newfuncgauss(/*double x[]*/double x, double y, double l, double z, double m, double n, double o, double p) {
    double sigma = 0.06;
    double mu = 0.0;
    double tsum = 0.0;
	  int ndim = 8;
    double k = sigma * sqrt(2.0 * M_PI);
    k = pow(k, ndim);
    k = 1.0 / k;
    //for (j = 1; j <= ndim; j++) 
	{
        tsum += (x - mu) * (x - mu) + 
				(y - mu) * (y - mu) + 
				(l - mu) * (l - mu) +
				(z - mu) * (z - mu) +
				(m - mu) * (m - mu) +
				(n - mu) * (n - mu) +
				(o - mu) * (o - mu) +
				(p - mu) * (p - mu);
    }
    tsum = tsum / (2 * sigma * sigma);
    tsum = exp(-tsum);
    return (tsum * k);
}

double funcbns(double x[]) {
    int i; double value = 0.0;
	int ndim = 8;
	double s = 4.0;
    for (i = 1; i <= ndim; i++) {
        value += x[i] * x[i];
    }
    return (pow(value, s / 2.0));
}

double funcsin(double x[]) {
    int i; double value;
	int ndim = 8;
    value = 0.0;
    for (i = 1; i <= ndim; i++) {
        value += x[i];
    }
    return (sin(value));
}

double funcrangesin(double x[]) {
    int i; double value;
	int ndim = 8;
    value = 0.0;
	double RANGESIN  = 10.0;        //upper limit for sin integral
	
    for (i = 1; i <= ndim; i++) {
        value += x[i];
    }
    return (pow(RANGESIN, ndim) * sin(RANGESIN * value));
}

template<int ndim, int N>
void printgrid(double x[ndim + 1][N + 1]) {
    int i, j;
    for (i = 1; i <= ndim; i++) {
        for (j = 0; j <= N; j++) {
            printf("%.3f ", x[i][j]);
        }
        printf("\n");
    }
}

template<typename F, int ndim>
void vegas(F integrand){
	constexpr int MXDIM  = 20;
	double ALPH = 1.5;
	int ITMAX = 10;
	double ITGRIDADJ = 0.;
	double TINY  = 1.0e-30;
	int N = 4096;      // N is number of slices
	//int ndim = 8;             // ndim is number of dimensions
	//int s = 4;                 // s for BNS(n,s) function
	
	
	
	 int i, j, k, l, i1, j1, iter, itcounter, npidx, idx, kk, npb;
    time_t t;
    double rc, xn, xnd, xo;
    double normconst, realden, realden2, newchi2, chi2 = 0.0;
    double LOWERIMITS[MXDIM + 1], UPPERLIMITS[MXDIM + 1], WIDTH[MXDIM + 1];
    double REGIONVOLUME = 1.0;
    double ISlice[ndim * ITMAX + 1][N + 1], VSlice[ndim * ITMAX + 1][N + 1];
    double IDim[ndim * ITMAX + 1], VDim[ndim * ITMAX + 1], w[ndim * ITMAX + 1], witer[ITMAX + 1];
    double lbj, ubj, lb, ub, wgtp, fp, fbp, f2p, f2bp, uwidth, iterinteg[ITMAX + 1], itervariance[ITMAX + 1];
    double IIter, VIter;
    double Sw, wsInt, wsIntSq, wtIntEst;
    double d[N + 1][MXDIM + 1], di[N + 1][MXDIM + 1], dt[MXDIM + 1], dx[MXDIM + 1];
    double r[N + 1], x[MXDIM + 1]/*, y[MXDIM + 1]*/, xi[MXDIM + 1][N + 1], xin[N + 1];
	std::array<double, MXDIM + 1> y;
	std::array<double, ndim> actual_y_array;
    /* seed the random number generator */
    //srand((unsigned) time(&t));

    /* initialize with uniform grid spacing */
    uwidth = 1.0 / N;
    for (i = 1; i <= ndim; i++) {
        for (j = 0; j <= N; j++) {
            xi[i][j] = j * uwidth;
        }
    }

// initializing limits
	
	double lows[] =  {20., 5.,  5., .15,  29., 0., 0.};
	double highs[] = {30., 50., 50.,.75,  38., 1., 6.28318530718};
	
    for (i = 1; i <= ndim; i++) {
        //LOWERIMITS[i]  = -1.0;
        //UPPERLIMITS[i] = 1.0;
		LOWERIMITS[i]  = lows[i-1];
        UPPERLIMITS[i] = highs[i-1];
        WIDTH[i] = (UPPERLIMITS[i] - LOWERIMITS[i]);
        REGIONVOLUME *= WIDTH[i];
    }

    chi2 = 0.0; Sw = 0.0; wsInt = 0.0; wtIntEst = 0.0; wsIntSq = 0.0;
    npb = 128;
    realden = ((double) N) * N * npb * npb * (npb - 1); //a divisor we need later for variance calculation
    realden2 = ((double)N) * npb;
    double inv_realden = 1.0/realden;
    double inv_realden2 = 1.0/realden2;
    itcounter = 0;
    printf("-----------------------------------------------------------------\n");
    printf("Starting with Uniformly Spaced Grid\n");
    printf("-----------------------------------------------------------------\n");

// set up for custom random number for testing
    double trn;
    uint64_t temp;
    uint32_t a = 1103515245;
    uint32_t c = 12345;
    uint32_t seed, seed_init;
    uint32_t one, expi;
    one = 1;
    expi = 31;
    uint32_t m = one << expi;
    int sliceId;

    double invndim = 1.0/ndim;
    for (iter = 1; iter <= ITMAX; iter++) {
        // if (iter <= ITGRIDADJ) {
        //     printf("Using iteration %d for grid adjustment\n", iter );
        //     printf("-----------------------------------------------------------------\n");
        // }
        sliceId = 0;
        seed_init = (iter) * ndim * N;
        iterinteg[iter] = 0.0;
        itervariance[iter] = 0.0;
        //if (iter > ITGRIDADJ) printf("%d\n", iter);
        for (i = 1; i <= ndim; i++) {
            itcounter += 1;
            IDim[itcounter] = 0.0;
            VDim[itcounter] = 0.0;
            for (j = 0; j < N; j++) {
                seed = seed_init + sliceId;
                sliceId = sliceId + 1;
                ISlice[itcounter][j + 1] = 0.0;
                VSlice[itcounter][j + 1] = 0.0;
                f2bp = fbp = 0.0;
                //fbp running sum of w(x)*f(x), f2bp running sum of (w(x)*f(x))^2
                // w(x)f(x) = N *(f(x)/p(x)) where p(x) is prob of picking x
                lbj = xi[i][j];
                ubj = xi[i][j + 1];
                for (npidx = 1; npidx <= npb; npidx++) {
                    wgtp = (ubj - lbj) * N; // wgtp will be w(x) when we finish picking x in each iteration
                    temp =  a * seed + c;
                    seed = temp & (m - 1);
                    trn = (double) seed / (double) m ;
                    //printf("inside trn:  %d  %d  %d   %e\n", iter, sliceId, npidx,  trn);
                    //x[i] = lbj + (rand() / (double)(RAND_MAX + 1.0)) * (ubj - lbj);
                    x[i] = lbj + trn * (ubj - lbj);
                    y[i] = LOWERIMITS[i] + WIDTH[i] * x[i];
                    for (k = i; k < i + ndim - 1; k++) {
                        kk = (k % ndim) + 1;
                        temp =  a * seed + c;
                        seed = temp & (m - 1);
                        trn = (double) seed / (double) m ;
                        idx = floor(trn * N);
                        lb = xi[kk][idx];
                        ub = xi[kk][idx + 1];
                        wgtp *= (ub - lb) * N;
                        temp =  a * seed + c;
                        seed = temp & (m - 1);
                        trn = (double) seed / (double) m ;
                        x[kk] = lb + trn * (ub - lb);
                        y[kk] = LOWERIMITS[kk] + WIDTH[kk] * x[kk];
						
                        // if ((i==5) && (j==38)) {
                        //     printf("kk, y  %d   %e\n", kk, y[kk]);
                        // }

                    }  // Here wgtp = w(x) = N^ndim*Vol_G(x) = N/p(x)
                    //fp = wgtp * REGIONVOLUME * newfuncgauss(y);

                    //fp =  inv_realden2 * invndim * wgtp * REGIONVOLUME * newfuncgauss(y);
					//fp =  inv_realden2 * invndim * wgtp * REGIONVOLUME * integrand(y);
					
					//temporary until MXDIM array size question is resolved
					for(int ii = 0; ii< MXDIM +1; ii++)
						actual_y_array[ii] = y[ii+1];
					
					fp =  inv_realden2 * invndim * wgtp * REGIONVOLUME * std::apply(integrand, actual_y_array);
			
                    f2p = fp * fp;
                    fbp += fp;
                    f2bp += f2p;
                    // if ((i==5) && (j==38)) {
                    //     printf("f   %.12e   fp   %.12e   f2p  %.12e   fbp  %.12e    f2bp  %.12e\n", funcgauss(y), fp, f2p, fbp, f2bp);
                    // }
                }
                // Here fbp = Sum w(x)f(x) = N*sum(f(x)/p(x))and f2bp = N^2 * sum((f(x)/p(x))^2)
                d[j + 1][i] = f2bp;
                //if (i == 5) printf("j, i, d   %d    %d    %e\n", j+1, i, d[j + 1][i]);
                ISlice[itcounter][j + 1] = fbp ;
                IDim[itcounter] += ISlice[itcounter][j + 1];
                // Calculations below are for the variance estimate of the integral estimate within the slice
                // math gives = 1/((npb)^2(npb-1) * N^2)[npb*f2bp - fbp^2]

                f2bp *= npb;
                f2bp = sqrt(f2bp);
                VSlice[itcounter][j + 1] = (f2bp - fbp) * (f2bp + fbp) ; //accomplishes the math,
                //realden is N^2(npb)^2(npb-1)
                if (VSlice[itcounter][j + 1] <= TINY)
                    VSlice[itcounter][j + 1] = TINY;
                VDim[itcounter] += VSlice[itcounter][j + 1];
            }

            iterinteg[iter] += IDim[itcounter];
            itervariance[iter] += VDim[itcounter];
        }

        //iterinteg[iter] = iterinteg[iter] / ndim;
        //iterinteg[iter] = iterinteg[iter] ;
        
        // itervariance[iter] = itervariance[iter] / (ndim * ndim);
        itervariance[iter] = itervariance[iter]/(npb-1.0) ;

        printf("iter =  %d    integral  = %e   variance = %e\n", iter, iterinteg[iter], itervariance[iter]);

//Calculate weighted integral estimate, weighted variance and chi2
        if (iter > ITGRIDADJ) {
            witer[iter] = 1.0 / itervariance[iter];
            Sw += witer[iter];          // Sw = Sum of 1/VDim[l] 1<=l<=i; weighted variance is inv of this
            wsInt += witer[iter] * iterinteg[iter];    // wsInt = sum w[l]*IDim[l] 1<=l<=itcounter
            wtIntEst = wsInt / Sw;      // weighted integral estimate up to current iteration
            wsIntSq += witer[iter] * iterinteg[iter] * iterinteg[iter]; // wsIntSq = sum w[l]*IDim[l]^2 1<=l<=i

            /* chi2/dof = (wsIntSq - wsInt*wtIntEst)/(itcounter-0.99999)*/
            if (iter < ITGRIDADJ + 1.5) {
                chi2 = 0.0;
            }
            else {
                chi2 = (wsIntSq - wsInt * wtIntEst) / (iter - ITGRIDADJ - 0.99999);
            }
            //for checking accuracy of chi2 calculation we calculate newchi2 directly
            newchi2 = 0.0;
            for (i1 = ITGRIDADJ + 1; i1 <= iter; i1++) {
                newchi2 += witer[i1] * (iterinteg[i1] - wtIntEst) * (iterinteg[i1] - wtIntEst);
            }
            newchi2 = newchi2 / (1.0 * itcounter - 0.99999);
           // printf ("%18.12f %18.12f %18.12f %12.4f \n", iterinteg[iter], wtIntEst, sqrt(1.0 / Sw), chi2);

        }

        // Grid Update should happen here
        // Copied Code from VEGAS
        for (j = 1; j <= ndim; j++) {
            xo = d[1][j];
            xn = d[2][j];
            d[1][j] = (xo + xn) / 2.0;
            dt[j] = d[1][j];
            for (i = 2; i < N; i++) {
                rc = xo + xn;
                xo = xn;
                xn = d[i + 1][j];
                d[i][j] = (rc + xn) / 3.0;
                dt[j] += d[i][j];
            }
            d[N][j] = (xo + xn) / 2.0;
            dt[j] += d[N][j];
            //printf("j , dt,  %d    %e\n", j , dt[j]);
        }
        for (j = 1; j <= ndim; j++) {
            rc = 0.0;
            for (i = 1; i <= N; i++) {
                if (d[i][j] < TINY) d[i][j] = TINY;
                r[i] = pow((1.0 - d[i][j] / dt[j]) /
                           (log(dt[j]) - log(d[i][j])), ALPH);
                //printf("i , r,  %d    %e\n", i , r[i]);
                rc += r[i];
            }
            rebin(rc / N, N, r, xin, xi[j]);
        }
        // if (iter >= ITGRIDADJ) {
        //     printf("-------------------------------------------------------------------------------------------\n");
        //     printf("Grid Updated\n");
        //     printf("-------------------------------------------------------------------------------------------\n");
        //     printf("it   Unwtd Integral     Wtd Integral          StdDev             Chi2\n");
        //     printf("--------------------------------------------------------------------------------------------\n");
        // }
    }

    IIter = wsInt / Sw;
    VIter = 1 / Sw;

// chi^2/dof = [1/(ndim - 1)]*Sum[witer[i]*(iterinteg[i] - IIter)^2]
    chi2 = 0.0;
    for (i = ITGRIDADJ + 1; i <= ITMAX; i++) {
        chi2 += witer[i] * (iterinteg[i] - IIter) * (iterinteg[i] - IIter);
    }
    chi2 = chi2 / (ITMAX - ITGRIDADJ - 0.99999);
    printf("Method: CheckRZU, ndim = %d , nslices = %d, ITMAX = %3d, ITGRIDADJ = %3d\n", ndim, N, ITMAX, ITGRIDADJ);
    printf("points/slice = %d\n", npb);
    printf("Integral       StandDev        chi2/dof  \n");
    printf("%e  %e %12.4f \n", IIter, sqrt(VIter), chi2);
}

template <typename F>
void
time_and_call_vegas(F f)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  auto t0 = std::chrono::high_resolution_clock::now();
  constexpr int ndim = 7;
  vegas<F, ndim>(f);
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
	
  std::cout<< dt.count() << std::endl;
}
