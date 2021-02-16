
/* Driver for routine vegas, shorter version

to avoid differnt cases */
#include <stdio.h>
//#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <ctime>
#include "cudaCuhre/quad/util/cudaApply.cuh"

#define WARP_SIZE 32
#define BLOCK_DIM_X 256
#define ALPH 1.5
#define NDMX  500
#define MXDIM 20
#define SKIP 0
#define NDMX1 NDMX+1
#define MXDIM1 MXDIM+1
//#define SCALE 1.0E+10
#define SCALE 1.0E-200
#define PI 3.14159265358979323846

#define IMAX(a,b) \
    ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a > _b ? _a : _b; })

#define IMIN(a,b) \
    ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a < _b ? _a : _b; })


//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
}
/*
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

 

#else

__device__ double atomicAdd(double* address, double val)

{

    unsigned long long int* address_as_ull = (unsigned long long int*)address;

    unsigned long long int old = *address_as_ull, assumed;

 

    do {

        assumed = old;

        old = atomicCAS(address_as_ull, assumed,

                        __double_as_longlong(val +

                               __longlong_as_double(assumed)));

 

    } while (assumed != old);

 

    return __longlong_as_double(old);

}

#endif
*/

__inline__ __device__
double warpReduceSum(double val) {
	val += __shfl_down_sync(0xffffffff, val, 16, WARP_SIZE);
	val += __shfl_down_sync(0xffffffff, val, 8, WARP_SIZE);
	val += __shfl_down_sync(0xffffffff, val, 4, WARP_SIZE);
	val += __shfl_down_sync(0xffffffff, val, 2, WARP_SIZE);
	val += __shfl_down_sync(0xffffffff, val, 1, WARP_SIZE);
	return val;
}

__inline__ __device__
double blockReduceSum(double val) {

	static __shared__ double shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);     // Each warp performs partial reduction

	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) val = warpReduceSum(val); //Final reduce within first warp

	return val;
}


__inline__ __device__  double fxn(double *xx) {
	// 6 d function
	double t = 0.0;
	for (int j = 1; j < 7; j++) {
		t += xx[j];
	}
	return sin(t);
	//return 0.01;
}


__inline__ __device__ double fxng(double x[]) {

	// gaussian function
	double sigma = 0.01;
	double tsum0 = 0.0; double k;
	double tsum1, tsum2;
	int j;
	int ndim = 9;
	k = sigma * sqrt(2.0 * M_PI);
	k = pow(k, ndim);
	k = 1.0 / k;
	for (j = 1; j <= ndim; j++) {
		tsum0 += (x[j]) * (x[j]);
	}
	tsum1 = tsum0 / (2 * sigma * sigma);
	tsum2 = exp(-tsum1);
	return (tsum2 * k);
}


#if 0
__inline__ __device__  double fxn3(double x[])
{

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
	return (tsum * k);
}

__inline__ __device__  double fxn7(double x[])
{

	//double sigma = 0.31622776601683794;
	double sigma = 0.01;
	double k;
	int j;
	k = (sigma * sqrt(2.0 * M_PI));
	k = pow(k, 9);
	k = 1.0 / k;

	double tsum = 0.0;
	for (j = 1; j < 10; j++) {
		tsum += (x[j]) * (x[j]);
	}
	tsum = -tsum / (2.0 * sigma * sigma);
	tsum = exp(tsum);
	return (tsum * k);
}

__inline__ __device__  double fxny(double *xx) {
	// 6 d function
	double t;
	double v, w, x, y, z;
	v = xx[1];
	w = xx[2];
	x = xx[3];
	y = xx[4];
	z = xx[5];
	t = 4.0 * v + 5.0 * w + 6.0 * x + 7.0 * y + 8.0 * z;
	return fabs(cos(t));
}

__inline__ __device__  double f2(double x[]) {
	int i; double value;
	value = 0.0;
	int ndim = 6;
	for (i = 1; i <= ndim; i++) {
		value += i * x[i];
	}
	return (cos(value));
}

__inline__ __device__  double f1(double x[]) {
	double value;
	value = 0.0;
	double a = 10.0;
	double b = 0.5;
	//int ndim = 1;
	value = 1 / pow(a, 2.0) + pow((x[1] - b), 2.0);
	value = 1 / value;
	return (value);
}

__inline__ __device__  double f3(double x[]) {
	int i; double value;
	value = 1.0;
	int ndim = 8;
	for (i = 1; i <= ndim; i++) {
		value += (ndim + 1 - i) * x[i];
	}
	value = pow(value, (ndim + 1));
	value = 1 / value;
	return (value);
}

__inline__ __device__  double f44(double x[]) {
	int i; double value;
	value = 0.0;
	//int ndim = 1;
	int ndim = 2;
	for (i = 1; i <= ndim; i++) {
		value += -(pow(25.0, 2) * pow((x[i] - 0.5), 2));
	}
	value = exp(value);
	return (value);
}

__inline__ __device__  double f4(double x[]) {
	int i; double value;
	value = 0.0;
	//int ndim = 1;
	int ndim = 2;
	for (i = 1; i <= ndim; i++) {
		value += (-10.0 * abs(x[i] - 0.5));
	}
	value = exp(value);
	return (value);
}

__inline__ __device__  double fxn4(double x[])
{

	double sigma = 0.31622776601683794;
	//double sigma = 0.02;
	double k;
	int j;
	k = (sigma * sqrt(2.0 * M_PI));
	k = pow(k, 9);
	k = 1.0 / k;
	int ndim = 9;
	double tsum = 0.0;
	for (j = 1; j <= ndim; j++) {
		tsum += (x[j]) * (x[j]);
	}
	tsum = -tsum / (2.0 * sigma * sigma);
	tsum = exp(tsum);
	return (tsum * k);
}

__inline__ __device__  double fxnT1(double *xx) {
	// 6 d function
	double t = 0.0;
	for (int j = 1; j < 7; j++) {
		t += xx[j];
	}
	//return sin(t);
	return 0.01;
}

__inline__ __device__  double fxnT(double x[])
{

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
	return (tsum * k);
}

#endif

__inline__ __device__  void get_indx(int ms, int *da, int ND, int NINTV) {
	int dp[MXDIM];
	int j, t0, t1;
	int m = ms;
	dp[0] = 1;
	dp[1] = NINTV;


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

__inline__ __device__  void get_indxN(int mc, int *da, int nd, int ng, double scc, double scic, double ing) {
	int kgt;
	for (int j = 0; j < nd - 1 ; j++) {
		kgt  = mc * scic ;
		mc = mc - kgt * scc ;
		scic = scic * ng;
		scc = scc * ing ;
		da[j] =  kgt + 1;
	}
	da[nd - 1] = mc + 1;

}

template <typename IntegT, int NDIM>
__global__ void vegas_kernel(IntegT* d_integrand, int ng, int npg, double xjac, double dxg,
                             double *result_dev, double xnd, double *xi,
                             double *d, double *dx, double *regn, int ncubes, 
                             int iter, double sc, double sci, double ing, 
                             int chunkSize, uint32_t totalNumThreads, int LastChunk) {

	uint64_t temp;
	uint32_t a = 1103515245;
	uint32_t c = 12345;
	uint32_t seed, seed_init;
	uint32_t one, expi;
	one = 1;
	expi = 31;
	uint32_t p = one << expi;

	seed_init = (iter) * ncubes;

	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;

	double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
	int kg[MXDIM + 1];
	int ia[MXDIM + 1];
	gpu::cudaArray<double, NDIM> x;
	int k, j;
	double fbg, f2bg;

	if (m < totalNumThreads) {
		if(m == totalNumThreads-1) chunkSize = LastChunk + 1;

		fbg = f2bg = 0.0;
		for(int t = 0; t<chunkSize; t++){
			fb = f2b = 0.0;
			get_indx(m*chunkSize+t, &kg[1], NDIM, ng);
			seed = seed_init + m*chunkSize + t;

			for ( k = 1; k <= npg; k++) {
				wgt = xjac;
				for ( j = 1; j <= NDIM; j++) {
					temp =  a * seed + c;
					seed = temp & (p - 1);
					ran00 = (double) seed / (double) p ;
			
					xn = (kg[j] - ran00) * dxg + 1.0;
					ia[j] = IMAX(IMIN((int)(xn), NDMX), 1);

					if (ia[j] > 1) {
						xo = xi[j * NDMX1 + ia[j]] - xi[j * NDMX1 + ia[j] - 1];
						rc = xi[j * NDMX1 + ia[j] - 1] + (xn - ia[j]) * xo;
					} else {
						xo = xi[j * NDMX1 + ia[j]];
						rc = (xn - ia[j]) * xo;
					}

					x[j-1] = regn[j] + rc * dx[j];
					wgt *= xo * xnd;
				}
				//double tmp = ((*fxn)(x));
                double tmp = gpu::apply(*d_integrand, x);
				f = wgt * tmp;
				f2 = f * f;
				fb += f;
				f2b += f2;

				for ( j = 1; j <= NDIM; j++) {
					atomicAdd(&d[ia[j]*MXDIM1 + j], fabs(f));
				}
			}  // end of npg loop

			f2b = sqrt(f2b * npg);
			f2b = (f2b - fb) * (f2b + fb);
			fbg += fb;
			f2bg += f2b;

		} //end of chunk for loop
			
		fbg  = blockReduceSum(fbg);
		f2bg = blockReduceSum(f2bg);

		if (tx == 0) {
			atomicAdd(&result_dev[0], fbg);
			atomicAdd(&result_dev[1], f2bg);
		}
	} // end of subcube if
}

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

template <typename IntegT, int NDIM>
void vegas(IntegT* d_integrand, double regn[], double (*fxn)(double [] ),
           double ncall, int itmx, int nprn, double *tgral, double *sd,
           double *chi2a)
{
	int i, it, j, k, nd, ndo, ng, npg, ncubes;
	//int ia[MXDIM + 1];
	double calls, dv2g, dxg, rc, ti, tsi, wgt, xjac, xn, xnd, xo;
	/* double d[(NDMX + 1)*(MXDIM + 1)], dt[MXDIM + 1],
	        dx[MXDIM + 1], r[NDMX + 1], x[MXDIM + 1], xi[(MXDIM + 1)*(NDMX + 1)], xin[NDMX + 1];*/

	double schi, si, swgt;
	double result[2];
	double *d, *dt, *dx, *r, *x, *xi, *xin;
	int *ia;

	d = (double*)malloc(sizeof(double) * (NDMX + 1) * (MXDIM + 1)) ;
	dt = (double*)malloc(sizeof(double) * (MXDIM + 1)) ;
	dx = (double*)malloc(sizeof(double) * (MXDIM + 1)) ;
	r = (double*)malloc(sizeof(double) * (NDMX + 1)) ;
	x = (double*)malloc(sizeof(double) * (MXDIM + 1)) ;
	xi = (double*)malloc(sizeof(double) * (MXDIM + 1) * (NDMX + 1)) ;
	xin = (double*)malloc(sizeof(double) * (NDMX + 1)) ;
	ia = (int*)malloc(sizeof(int) * (MXDIM + 1)) ;

// code works only  for (2 * ng - NDMX) >= 0)

	ndo = 1;
	for (j = 1; j <= NDIM; j++) xi[j * NDMX1 + 1] = 1.0;
	si = swgt = schi = 0.0;
	nd = NDMX;
	ng = 1;
	ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / NDIM);
	for (k = 1, i = 1; i < NDIM; i++) k *= ng;
	double sci = 1.0 / k;
	double sc = k;
	k *= ng;
	ncubes = k;
	npg = IMAX(ncall / k, 2);
	calls = (double)npg * (double)k;
	dxg = 1.0 / ng;
	double ing = dxg;
	for (dv2g = 1, i = 1; i <= NDIM; i++) dv2g *= dxg;
	dv2g = (calls * dv2g * calls * dv2g) / npg / npg / (npg - 1.0);
	xnd = nd;
	dxg *= xnd;
	xjac = 1.0 / calls;
	for (j = 1; j <= NDIM; j++) {
		dx[j] = regn[j + NDIM] - regn[j];
		//printf("%e, %e\n", dx[j], xjac);
		xjac *= dx[j];
	}

	for (i = 1; i <= IMAX(nd, ndo); i++) r[i] = 1.0;
	for (j = 1; j <= NDIM; j++) rebin(ndo / xnd, nd, r, xin, &xi[j * NDMX1]);
	ndo = nd;

	printf("ng, npg, ncubes, xjac, %d, %d, %12d, %e\n", ng, npg, ncubes, xjac);
    
	double *d_dev, *dx_dev, *x_dev, *xi_dev, *regn_dev,  *result_dev;
	int *ia_dev;

	cudaMalloc((void**)&result_dev, sizeof(double) * 2); cudaCheckError();
	cudaMalloc((void**)&d_dev, sizeof(double) * (NDMX + 1) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&dx_dev, sizeof(double) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&x_dev, sizeof(double) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&xi_dev, sizeof(double) * (MXDIM + 1) * (NDMX + 1)); cudaCheckError();
	cudaMalloc((void**)&regn_dev, sizeof(double) * ((NDIM * 2) + 1)); cudaCheckError();
	cudaMalloc((void**)&ia_dev, sizeof(int) * (MXDIM + 1)); cudaCheckError();

	cudaMemcpy( dx_dev, dx, sizeof(double) * (MXDIM + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
	cudaMemcpy( x_dev, x, sizeof(double) * (MXDIM + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
	cudaMemcpy( regn_dev, regn, sizeof(double) * ((NDIM * 2) + 1), cudaMemcpyHostToDevice) ; cudaCheckError();

	cudaMemset(ia_dev, 0, sizeof(int) * (MXDIM + 1));
	
	int chunkSize = 2048;
	uint32_t totalNumThreads = (uint32_t) ((ncubes + chunkSize -1) / chunkSize); 
	uint32_t totalCubes = totalNumThreads * chunkSize;
	int extra = totalCubes - ncubes;
	int LastChunk = chunkSize - extra;
	uint32_t nBlocks = ((uint32_t) (((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X))/chunkSize) + 1;
	uint32_t nThreads = BLOCK_DIM_X;
    
	printf("ncubes %d nBlocks %d nThreads %d totalNumThreads %d totalCubes %d extra  %d LastChunk %d\n", ncubes, nBlocks, nThreads, totalNumThreads, totalCubes, extra, LastChunk);
	printf("the number of evaluation will be %e\n", calls);
    
	for (it = 1; it <= itmx; it++) {

		ti = tsi = 0.0;
		for (j = 1; j <= NDIM; j++) {
			for (i = 1; i <= nd; i++) d[i * MXDIM1 + j] = 0.0;
		}

		cudaMemcpy( xi_dev, xi, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
		cudaMemset(d_dev, 0, sizeof(double) * (NDMX + 1) * (MXDIM + 1));
		cudaMemset(result_dev, 0, 2 * sizeof(double));

		vegas_kernel<IntegT, NDIM><<< nBlocks, nThreads>>>(d_integrand, ng, npg, xjac, dxg, result_dev, xnd,
		                                      xi_dev, d_dev, dx_dev, regn_dev, ncubes, it, sc, sci,  ing, chunkSize, totalNumThreads, LastChunk);

		cudaMemcpy(xi, xi_dev, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyDeviceToHost); cudaCheckError();
		cudaMemcpy( d, d_dev,  sizeof(double) * (NDMX + 1) * (MXDIM + 1), cudaMemcpyDeviceToHost) ; cudaCheckError();

		cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);

		//printf("ti is %f", ti);
		ti  = result[0];
		tsi = result[1];
		tsi *= dv2g;
		printf("iter = %d  integ = %e   std = %e\n", it, ti, sqrt(tsi));

		if (it > SKIP) {
			wgt = 1.0 / tsi;
			si += wgt * ti;
			schi += wgt * ti * ti;
			swgt += wgt;
			*tgral = si / swgt;
			*chi2a = (schi - si * (*tgral)) / (it - 0.9999);
			if (*chi2a < 0.0) *chi2a = 0.0;
			*sd = sqrt(1.0 / swgt);
			tsi = sqrt(tsi);
			//printf("it %d\n", it);
			printf("%5d   %14.7g+/-%9.2g  %9.2g\n", it, *tgral, *sd, *chi2a);
		}
		//printf("%3d   %e  %e\n", it, ti, tsi);

		for (j = 1; j <= NDIM; j++) {
			xo = d[1 * MXDIM1 + j];
			xn = d[2 * MXDIM1 + j];
			d[1 * MXDIM1 + j] = (xo + xn) / 2.0;
			dt[j] = d[1 * MXDIM1 + j];
			for (i = 2; i < nd; i++) {
				rc = xo + xn;
				xo = xn;
				xn = d[(i + 1) * MXDIM1 + j];
				d[i * MXDIM1 + j] = (rc + xn) / 3.0;
				dt[j] += d[i * MXDIM1 + j];
			}
			d[nd * MXDIM1 + j] = (xo + xn) / 2.0;
			dt[j] += d[nd * MXDIM1 + j];
			//printf("iter, j, dtj:    %d    %d      %e\n", it, j, dt[j]);
		}

		for (j = 1; j <= NDIM; j++) {
			if (dt[j] > 0.0) {
				rc = 0.0;
				for (i = 1; i <= nd; i++) {
					//if (d[i * MXDIM1 + j] < TINY) d[i * MXDIM1 + j] = TINY;
					r[i] = pow((1.0 - d[i * MXDIM1 + j] / dt[j]) /
					           (log(dt[j]) - log(d[i * MXDIM1 + j])), ALPH);
					rc += r[i];
				}

				rebin(rc / xnd, nd, r, xin, &xi[j * NDMX1]);
			}
		}
	}

	free(d);
	free(dt);
	free(dx);
	free(ia);
	free(x);
	free(xi);

	cudaFree(d_dev);
	cudaFree(dx_dev);
	cudaFree(ia_dev);
	cudaFree(x_dev);
	cudaFree(xi_dev);
	cudaFree(regn_dev);

}

template <typename IntegT, int NDIM>
int Vegas_mcubes(IntegT integrand)
{
    IntegT* d_integrand;
    cudaMallocManaged((void**)&d_integrand, sizeof(IntegT));
    memcpy(d_integrand, &integrand, sizeof(IntegT));
    
	int itmax, j, nprn, ncall;
	double avgi, chi2a, sd;
	double regn[2 * MXDIM + 1];

	ncall = 2000000000;
	itmax = 59;
	nprn = -1;
	avgi = sd = chi2a = 0.0;
    
	/*for (j = 1; j <= NDIM; j++) {
		regn[j] = 0.0;
		regn[j + NDIM] = 10.0;
	}*/
    
    regn[1] = 20.;
    regn[2] = 5.;
    regn[3] = 5.;
    regn[4] = .15;
    regn[5] = 29.;
    regn[6] = 0.;
    regn[7] = 0.;
    // high bounds
    regn[8] = 30.;
    regn[9] = 50.;
    regn[10] = 50.;
    regn[11] = .75;
    regn[12] = 38.;
    regn[13] = 1.;
    regn[14] = 6.28318530718;
    
	vegas<IntegT, NDIM>(d_integrand, regn, fxn, ncall, itmax, nprn, &avgi, &sd, &chi2a);
	printf("Number of iterations performed: %d\n", itmax);
	printf("Integral, Standard Dev., Chi-sq. = %.18f %.20f% 12.6f\n", avgi, sd, chi2a);
	return 0;

}


