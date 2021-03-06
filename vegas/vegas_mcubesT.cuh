/*

code works for gaussian and sin using switch statement. device pointerr/template slow
down the code by 2x

chunksize needs to be tuned based on the ncalls. For now hardwired using a switch statement

OR
nvcc -O2 -DCURAND -o vegas_mcubes vegas_mcubes.cu -arch=sm_70

example run command

nvprof ./vegas_mcubes 0 6 0.0  10.0  2.0E+09  10, 5, 0

Last three arguments are: total iterations, iteration

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <ctime>

#define WARP_SIZE 32
#define BLOCK_DIM_X 128
#define ALPH 1.5
#define NDMX  500 //does 500 splits on each axis
#define MXDIM 20

#define NDMX1 NDMX+1
#define MXDIM1 MXDIM+1
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

__inline__ __device__  double func1(double *xx, int ndim) {
	// 6 d function
	double t = 0.0;
	for (int j = 1; j <= ndim; j++) {
		t += xx[j];
	}
	return sin(t);
}


__inline__ __device__ double func2(double x[], int ndim) {
	// gaussian function
	double sigma = 0.01;
	double tsum0 = 0.0; double k;
	double tsum1, tsum2;
	int j;
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


__inline__ __device__ double func3(double x[], int ndim){

    //if(u > 0.1 || v > 0.1)
   //     printf("%f, %f\n", u, v);
    double k = 0.01890022674239546529975841;
    double u = x[1];
    double v = x[2];
    return 4*k*u*u/(.01 + pow(u-v-(1./3.),2));
  
}
//#include "func.cuh"

__inline__ __device__ double BoxIntegral8_22(double x[], int ndim){
 
    double s = 22;
    double sum = 0;
    for(int i=1; i<= ndim; i++){
        sum += pow(x[i], 2); 
    }
	return pow(sum, s / 2);
  
}

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

__inline__ __device__  void get_indxT(int mc, int *da, int nd, int ng, double scc, double scic, double ing) {
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

__global__ void vegas_kernel(int ng, int ndim, int npg, double xjac, double dxg,
                             double *result_dev, double xnd, double *xi,
                             double *d, double *dx, double *regn, int ncubes,
                             int iter, double sc, double sci, double ing,
                             int chunkSize, uint32_t totalNumThreads,
                             int LastChunk, int fcode) {


#ifdef CUSTOM
	uint64_t temp;
	uint32_t a = 1103515245;
	uint32_t c = 12345;
	uint32_t one, expi;
	one = 1;
	expi = 31;
	uint32_t p = one << expi;
#endif


	uint32_t seed, seed_init;
	seed_init = (iter) * ncubes;



	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;


	double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
	int kg[MXDIM + 1];
	int ia[MXDIM + 1];
	double x[MXDIM + 1];
	int k, j;
	double fbg, f2bg;

	if (m < totalNumThreads) {
		if (m == totalNumThreads - 1) chunkSize = LastChunk + 1;
		//if(tx == 30 && blockIdx.x == 6771) printf("here m is %d\n", m);
		seed = seed_init + m * chunkSize;
#ifdef CURAND
		curandState localState;
		curand_init(seed, 0, 0, &localState);
#endif
		fbg = f2bg = 0.0;
		get_indx(m * chunkSize, &kg[1], ndim, ng);
		for (int t = 0; t < chunkSize; t++) {
			fb = f2b = 0.0;
			//get_indx(m * chunkSize + t, &kg[1], ndim, ng);

			for ( k = 1; k <= npg; k++) {
				wgt = xjac;

				for ( j = 1; j <= ndim; j++) {
#ifdef CUSTOM
					temp =  a * seed + c;
					seed = temp & (p - 1);
					ran00 = (double) seed / (double) p ;
#endif
#ifdef CURAND
					ran00 = curand_uniform(&localState);
#endif

					xn = (kg[j] - ran00) * dxg + 1.0;
					ia[j] = IMAX(IMIN((int)(xn), NDMX), 1);

					if (ia[j] > 1) {
						xo = xi[j * NDMX1 + ia[j]] - xi[j * NDMX1 + ia[j] - 1];
						rc = xi[j * NDMX1 + ia[j] - 1] + (xn - ia[j]) * xo;
					} else {
						xo = xi[j * NDMX1 + ia[j]];
						rc = (xn - ia[j]) * xo;
					}

					x[j] = regn[j] + rc * dx[j];
                               
					wgt *= xo * xnd;


				}
				//double tmp = func[1](x, ndim);
				double tmp;

				switch (fcode) {
				case 0:
					tmp = (*func1)(x, ndim);
					break;
				case 1:
					tmp = (*func2)(x, ndim);
					break;
				case 2:
					tmp = (*func3)(x, ndim);
					break;					
                case 3: 
                    tmp = (*BoxIntegral8_22)(x, ndim);
                    break;
				default:
					tmp = (*func2)(x, ndim);
					break;
				}

//        tmp = (*func2)(x, ndim);
				f = wgt * tmp;
				f2 = f * f;

				fb += f;
				f2b += f2;
#pragma unroll 2
				for ( j = 1; j <= ndim; j++) {
					atomicAdd(&d[ia[j]*MXDIM1 + j], fabs(f));
					//d[ia[j]*MXDIM1 + j] += f2;
				}

			}  // end of npg loop

			f2b = sqrt(f2b * npg);
			f2b = (f2b - fb) * (f2b + fb);

			fbg += fb;
			f2bg += f2b;

			for (int k = ndim; k >= 1; k--) {
				kg[k] %= ng;
				if (++kg[k] != 1) break;
			}

		} //end of chunk for loop

		fbg  = blockReduceSum(fbg);
		f2bg = blockReduceSum(f2bg);
        
        if(threadIdx.x == 0 && blockIdx.x == 0)
            printf("block %i: %e +- %e\n", blockIdx.x, fbg, f2bg); 

		if (tx == 0) {
			atomicAdd(&result_dev[0], fbg);
			atomicAdd(&result_dev[1], f2bg);
            //printf("ti is %e, tsi is %e\n", result_dev[0], result_dev[1]);

		}


	} // end of subcube if

}

__global__ void vegas_kernelF(int ng, int ndim, int npg, double xjac, double dxg,
                              double *result_dev, double xnd, double *xi,
                              double *d, double *dx, double *regn, int ncubes,
                              int iter, double sc, double sci, double ing,
                              int chunkSize, uint32_t totalNumThreads,
                              int LastChunk, int fcode) {

#ifdef CUSTOM
	uint64_t temp;
	uint32_t a = 1103515245;
	uint32_t c = 12345;
	uint32_t one, expi;
	one = 1;
	expi = 31;
	uint32_t p = one << expi;
#endif
    
	uint32_t seed, seed_init;
	seed_init = (iter) * ncubes;

	int m = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;

	double fb, f2b, wgt, xn, xo, rc, f, f2, ran00;
	int kg[MXDIM + 1];
	//int ia[MXDIM + 1];
	int iaj;
	double x[MXDIM + 1];
	int k, j;
	double fbg, f2bg;
	//if(tx == 30 && blockIdx.x == 6771) printf("here m is %d\n", m);

	if (m < totalNumThreads) {
		if (m == totalNumThreads - 1) chunkSize = LastChunk + 1;
		//if(tx == 30 && blockIdx.x == 6771) printf("here m is %d\n", m);
		seed = seed_init + m * chunkSize;
#ifdef CURAND
		curandState localState;
		curand_init(seed, 0, 0, &localState);
#endif
		fbg = f2bg = 0.0;
		get_indx(m * chunkSize, &kg[1], ndim, ng);
		for (int t = 0; t < chunkSize; t++) {
			fb = f2b = 0.0;
			//get_indx(m * chunkSize + t, &kg[1], ndim, ng);

			for ( k = 1; k <= npg; k++) {
				wgt = xjac;

				for ( j = 1; j <= ndim; j++) {
#ifdef CUSTOM
					temp =  a * seed + c;
					seed = temp & (p - 1);
					ran00 = (double) seed / (double) p ;
#endif
#ifdef CURAND
					ran00 = curand_uniform(&localState);
#endif

					xn = (kg[j] - ran00) * dxg + 1.0;
					iaj = IMAX(IMIN((int)(xn), NDMX), 1);

					if (iaj > 1) {
						xo = xi[j * NDMX1 + iaj] - xi[j * NDMX1 + iaj - 1];
						rc = xi[j * NDMX1 + iaj - 1] + (xn - iaj) * xo;
					} else {
						xo = xi[j * NDMX1 + iaj];
						rc = (xn - iaj) * xo;
					}

					//x[j] = regn[j] + rc * dx[j];

					x[j] = regn[1] + rc * dx[1];

					wgt *= xo * xnd;

				}
				//double tmp = func[1](x, ndim);
				double tmp;

				switch (fcode) {
				case 0:
					tmp = (*func1)(x, ndim);
					break;
				case 1:
					tmp = (*func2)(x, ndim);
					break;
				case 2:
					tmp = (*func3)(x, ndim);
					break;
                case 3: 
                    tmp = (*BoxIntegral8_22)(x, ndim);
                    break;
				default:
					tmp = (*func2)(x, ndim);
					break;
				}

//        tmp = (*func2)(x, ndim);
				f = wgt * tmp;
				f2 = f * f;

				fb += f;
				f2b += f2;


			}  // end of npg loop

			f2b = sqrt(f2b * npg);
			f2b = (f2b - fb) * (f2b + fb);

			fbg += fb;
			f2bg += f2b;

			for (int k = ndim; k >= 1; k--) {
				kg[k] %= ng;
				if (++kg[k] != 1) break;
			}

		} //end of chunk for loop

		fbg  = blockReduceSum(fbg);
		f2bg = blockReduceSum(f2bg);


		if (tx == 0) {
			atomicAdd(&result_dev[0], fbg);
			atomicAdd(&result_dev[1], f2bg);
			//printf("ti is %e, tsi is %e\n",result_dev[0], result_dev[1]);

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

}

void vegas(double regn[], int ndim, int fcode,
           double ncall, double *tgral, double *sd,
           double *chi2a, int titer, int itmax, int skip, int chunkSIZE)

{

	int i, it, j, k, nd, ndo, ng, npg, ncubes;
	double calls, dv2g, dxg, rc, ti, tsi, wgt, xjac, xn, xnd, xo;

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
    
	for (j = 1; j <= ndim; j++) 
        xi[j * NDMX1 + 1] = 1.0;
        
	si = swgt = schi = 0.0;
	nd = NDMX;
	ng = 1;
	ng = (int)pow(ncall / 2.0 + 0.25, 1.0 / ndim);
    
	for (k = 1, i = 1; i < ndim; i++) 
        k *= ng;
        
	double sci = 1.0 / k;
	double sc = k;
	k *= ng;
	ncubes = k;
    
    //std::cout<<"ncubes:"<<ncubes<<std::endl;
    
	npg = IMAX(ncall / k, 2);
	calls = (double)npg * (double)k;
	dxg = 1.0 / ng;
	double ing = dxg;
    
	for (dv2g = 1, i = 1; i <= ndim; i++) 
        dv2g *= dxg;
        
	dv2g = (calls * dv2g * calls * dv2g) / npg / npg / (npg - 1.0);
	xnd = nd;
	dxg *= xnd;
	xjac = 1.0 / calls;
    
	for (j = 1; j <= ndim; j++) {
		dx[j] = regn[j + ndim] - regn[j];
		xjac *= dx[j];
	}

	for (i = 1; i <= IMAX(nd, ndo); i++) 
        r[i] = 1.0;
	for (j = 1; j <= ndim; j++) 
        rebin(ndo / xnd, nd, r, xin, &xi[j * NDMX1]);
        
	ndo = nd;

	double *d_dev, *dx_dev, *x_dev, *xi_dev, *regn_dev,  *result_dev;
	int *ia_dev;

	cudaMalloc((void**)&result_dev, sizeof(double) * 2); cudaCheckError();
	cudaMalloc((void**)&d_dev, sizeof(double) * (NDMX + 1) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&dx_dev, sizeof(double) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&x_dev, sizeof(double) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&xi_dev, sizeof(double) * (MXDIM + 1) * (NDMX + 1)); cudaCheckError();
	cudaMalloc((void**)&regn_dev, sizeof(double) * ((ndim * 2) + 1)); cudaCheckError();
	cudaMalloc((void**)&ia_dev, sizeof(int) * (MXDIM + 1)); cudaCheckError();

	cudaMemcpy( dx_dev, dx, sizeof(double) * (MXDIM + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
	cudaMemcpy( x_dev, x, sizeof(double) * (MXDIM + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
	cudaMemcpy( regn_dev, regn, sizeof(double) * ((ndim * 2) + 1), cudaMemcpyHostToDevice) ; cudaCheckError();

	cudaMemset(ia_dev, 0, sizeof(int) * (MXDIM + 1));

	int chunkSize = chunkSIZE;

	/*switch (fcode) {
	case 0:
		chunkSize = 2048;
		break;
	case 1:
		chunkSize = 32;
		break;
	case 2:
		chunkSize = 2048;
		break;		
	default:
		chunkSize = 32;
		break;
	}*/


	uint32_t totalNumThreads = (uint32_t) ((ncubes + chunkSize - 1) / chunkSize);
	uint32_t totalCubes = totalNumThreads * chunkSize;
	int extra = totalCubes - ncubes;
	int LastChunk = chunkSize - extra;
	uint32_t nBlocks = ((uint32_t) (((ncubes + BLOCK_DIM_X - 1) / BLOCK_DIM_X)) / chunkSize) + 1;
	uint32_t nThreads = BLOCK_DIM_X;
	/*printf("ncubes %d nBlocks %d nThreads %d totalNumThreads %d totalCubes %d extra  %d LastChunk %d\n",
	       ncubes, nBlocks, nThreads, totalNumThreads, totalCubes, extra, LastChunk);*/

	//printf("the number of evaluation will be %e\n", calls);
	for (it = 1; it <= itmax; it++) {

		ti = tsi = 0.0;
		for (j = 1; j <= ndim; j++) {
			for (i = 1; i <= nd; i++) 
                d[i * MXDIM1 + j] = 0.0;
		}


		cudaMemcpy( xi_dev, xi, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
		cudaMemset(d_dev, 0, sizeof(double) * (NDMX + 1) * (MXDIM + 1));
		cudaMemset(result_dev, 0, 2 * sizeof(double));
        
        //std::cout<<"Launching kernel with:"<<nBlocks<<","<<nThreads<<std::endl;
        
		vegas_kernel <<< nBlocks, nThreads>>>(ng, ndim, npg, xjac, dxg, result_dev, xnd,
		                                      xi_dev, d_dev, dx_dev, regn_dev, ncubes, it, sc,
		                                      sci,  ing, chunkSize, totalNumThreads,
		                                      LastChunk, fcode);


		cudaMemcpy(xi, xi_dev, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyDeviceToHost); cudaCheckError();
		cudaMemcpy(d, d_dev,  sizeof(double) * (NDMX + 1) * (MXDIM + 1), cudaMemcpyDeviceToHost) ; cudaCheckError();

		cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);

		//printf("ti is %f", ti);
		ti  = result[0];
		tsi = result[1];
		tsi *= dv2g;
        
		printf("iter = %d  integ = %e   std = %e\n", it, ti, sqrt(tsi));

		if (it > skip) {
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
			//printf("%5d   %14.7g+/-%9.2g  %9.2g\n", it, *tgral, *sd, *chi2a);
		}
		//printf("%3d   %e  %e\n", it, ti, tsi);
        
		for (j = 1; j <= ndim; j++) {
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

		for (j = 1; j <= ndim; j++) {
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

	}  // end of iterations

	//  Start of iterations without adjustment

	cudaMemcpy( xi_dev, xi, sizeof(double) * (MXDIM + 1) * (NDMX + 1), cudaMemcpyHostToDevice) ; cudaCheckError();

	for (it = itmax+1; it <= titer; it++) {

		ti = tsi = 0.0;

		cudaMemset(result_dev, 0, 2 * sizeof(double));

		vegas_kernelF <<< nBlocks, nThreads>>>(ng, ndim, npg, xjac, dxg, result_dev, xnd,
		                                       xi_dev, d_dev, dx_dev, regn_dev, ncubes, it, sc,
		                                       sci,  ing, chunkSize, totalNumThreads,
		                                       LastChunk, fcode);


		cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);

		//printf("ti is %f", ti);
		ti  = result[0];
		tsi = result[1];
		tsi *= dv2g;
		//printf("iter = %d  integ = %e   std = %e\n", it, ti, sqrt(tsi));
        //printf("%d, %.16e, %.16e\n", it, funcevals, chunkSize, ti, sqrt(tsi));
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
		//printf("%5d   %14.7g+/-%9.4g  %9.2g\n", it, *tgral, *sd, *chi2a);
		//printf("%3d   %e  %e\n", it, ti, tsi);

	}  // end of iterations

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

//int main(int argc, char **argv)
void vegas_mcubes(int functionID, int NDIM, double* lbound, double* rbound, double _ncall, int minIters, int maxIters, int _skip, int chunkSIZE)
{
	/*if (argc < 9) {
		printf( "****************************************\n"
		        "Usage (6 arguments):\n"
		        "./vegas_mcubes FCODE  DIM LL  UL  NCALLS  SKIP\n"
		        "FCODE = 0 to MAX_NUMBER_OF_FUNCTIONS-1\n"
		        "NCALLS in scientific notation, e.g. 1.0E+07 \n"
		        "****************************************\n");
		exit(-1);
	}*/
	int  j;
	double avgi, chi2a, sd;
	double regn[2 * MXDIM + 1];

	/*int fcode = atoi(argv[1]);
	int ndim = atoi(argv[2]);
	float LL = atof(argv[3]);
	float UL = atof(argv[4]);
	double ncall = atof(argv[5]);
	int titer = atoi(argv[6]);
	int itmax = atoi(argv[7]);
	int skip = atoi(argv[8]);*/
    
    int fcode = functionID;
	int ndim = NDIM;
	//float LL = atof(argv[3]);
	//float UL = atof(argv[4]);
	double ncall = _ncall;
	int titer = minIters;
	int itmax = maxIters;
	int skip = _skip;

	avgi = sd = chi2a = 0.0;
	for (j = 1; j <= ndim; j++) {
		regn[j] = lbound[j-1];
		regn[j + ndim] = rbound[j-1];
	}


	vegas(regn, ndim, fcode, ncall, &avgi, &sd, &chi2a, titer, itmax, skip, chunkSIZE);

	//printf("Number of iterations performed: %d\n", itmax);

	//printf("%.16e, %.16e, %.16e,\t", avgi, sd, chi2a);
}


