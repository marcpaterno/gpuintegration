// this version handles multiple iterations where each iteration handles ndim dimensions
// The stopping criteria with the rel_err added
// It prints out a weighted integral estimate, variance, std.dev and chi2/dof after each dim
// of each iteration
// For this purpose we maintain several running sums over iterations
// (i) SInt = running sum of individual integral estimates -- not here
// (ii) Sw = running sum of 1/Var[l] (also called weights)
// (iii) Running Sum of squares of individual estimates -- not in this version yet

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>
#include <math.h>
#include <curand_kernel.h>


#define MXDIM 10
#define MXDIM1 11
#define ALPH 1.5
#define ITMAX 96 // number of iterations to be performed        
#define TINY 1.0e-30
#define SKIP 85
//#define N 1024
//#define N1 1025           // N is number of slices
//#define ndim 6                // ndim is number of dimensions
#define s 4                 // s for BNS(n,s) function
#define RANGESIN 3.0        //upper limit for sin integral
#define WARP_SIZE 32



//void printgrid(double y[ndim+1][N+1]);

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
}

void printarray(double *x, int K) {
	for (int i = 0; i < K - 1; i++) {
		printf("i = %d,    value = %e\n", i, x[i]);

	}
}


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

__inline__ __device__  double newfuncgauss(double x[], int ndim){
    //double sigma = 0.06;
    double sigma = 0.013;
    double mu = 0.0; 
    double tsum = 0.0; double k;
    int j;
    
    k = sigma * sqrt(2.0*M_PI);
    k = pow(k, ndim);
    k = 1.0 / k;
    for (j = 1; j <= ndim; j++) {
        tsum += (x[j] - mu) * (x[j] - mu);
    }
    tsum = tsum/(2*sigma*sigma);
    if(tsum<80.0)
    	tsum = exp(-tsum);
    else tsum = TINY;
    return (tsum * k);
}


__inline__ __device__  double fxn(double x[], int ndim)

{

	double sigma = 0.31622776601683794;
	//double sigma = 0.02;
	double k;
	int j;
	k = (sigma * sqrt(2.0 * M_PI));
	k = pow(k, 9);
	k = 1.0 / k;

	double tsum = 0.0;
	for (j = 1; j <= ndim; j++) {
		tsum += (x[j]) * (x[j]);
	}
	tsum = -tsum / (2.0 * sigma * sigma);
	tsum = exp(tsum);
	return (tsum * k);
}

__inline__ __device__ double funcgauss(double x[], int ndim) {
	double sigma = 0.03;
	double mu = 0.5;
	double tsum = 0.0; double k;
	int j;
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

__inline__ __device__  double funcsin(double x[], int ndim) {
	int i; double value;
	value = 0.0;
	for (i = 1; i <= ndim; i++) {
		value += x[i];
	}
	return (sin(value));
}

__inline__ __device__  double f1(double x[], int ndim) {
	int i; double value;
	value = 0.0;
	for (i = 1; i <= ndim; i++) {
		value += i * x[i];
	}
	return (cos(value));
}

__inline__ __device__  double f2(double x[], int ndim) {
	double value;
	value = 0.0;
	double a = 10.0;
	double b = 0.5;
	value = 1 / pow(a, 2.0) + pow((x[1] - b), 2.0);
	value = 1 / value;
	return (value);
}

__inline__ __device__  double f3(double x[], int ndim) {
	int i; double value;
	value = 1.0;
	for (i = 1; i <= ndim; i++) {
		value += (ndim + 1 - i) * x[i];
	}
	value = pow(value, (ndim + 1));
	value = 1 / value;
	return (value);
}

__inline__ __device__  double f4(double x[], int ndim) {
	int i; double value;
	value = 0.0;
	for (i = 1; i <= ndim; i++) {
		value += -(pow(25.0, 2) * pow((x[i] - 0.5), 2));
	}
	value = exp(value);
	return (value);
}

__inline__ __device__  double f5(double x[], int ndim) {
	int i; double value;
	value = 0.0;
	for (i = 1; i <= ndim; i++) {
		value += (-10.0 * abs(x[i] - 0.5));
	}
	value = exp(value);
	return (value);
}

__inline__ __device__
double warpReduceSum(double val) {
	val += __shfl_down_sync(0xffffffff, val, 16, WARP_SIZE);
	val += __shfl_down_sync(0xffffffff, val, 8,  WARP_SIZE);
	val += __shfl_down_sync(0xffffffff, val, 4,  WARP_SIZE);
	val += __shfl_down_sync(0xffffffff, val, 2,  WARP_SIZE);
	val += __shfl_down_sync(0xffffffff, val, 1,  WARP_SIZE);
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

template <typename IntegT, int NDIM>
__global__ void vegas(int it, double *dx, double xjacd, double *regn,
                      double *result_dev, int totalSliceNum, int npb, double xi[],
                      double d[], int N, int N1, int ndim, int iter) {

	int sliceId, i, j, npidx, kk, k, idx;
	double f2bd, slice_integ;
	double f, fd, f2d;
	double lbj, ubj, lb, ub, wgtp;
	double x[MXDIM + 1];
	double trn;
	
	sliceId = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;

	curandState localState;
	// need to revisit whether sliceID to be used in second argument to improve random numebr quality
	curand_init((unsigned long long)clock() + sliceId, 0, 0, &localState);
	//for(sliceId = 0; sliceId<totalSliceNum; sliceId++){
	if (sliceId < totalSliceNum) {
		i = (sliceId / N) + 1;
		j = sliceId % N;
		//seed = seed_init + sliceId;
		slice_integ = 0.0;
		f2bd = 0.0;
		lbj = xi[i * N1 + j];
		ubj = xi[i * N1 + j + 1];
		for (npidx = 0; npidx < npb; npidx++) {
			wgtp = (ubj - lbj) * N * xjacd; // wgtp will be w(x) when we finish picking x in each iteration
			//printf("inside trn:  %d  %d  %d   %e\n", it, sliceId, npidx,  trn);
			trn = (ubj - lbj) * curand_uniform(&localState) + lbj;
			x[i] = regn[i] + trn * dx[i]; //*(ubj-lbj);
			for (k = i; k < i + ndim - 1; k++) {
				kk = (k % ndim) + 1;
				trn = curand_uniform(&localState);
				idx = int(trn * N);
				lb = xi[kk * N1 + idx];
				ub = xi[kk * N1 + idx + 1];
				wgtp *= (ub - lb) * N;
				trn = (ub - lb) * curand_uniform(&localState) + (lb);
				x[kk] = regn[i] + trn * dx[i]; //*(ub-lb)

			}
			f = newfuncgauss(x, ndim);
			fd = wgtp * f;
			f2d = fd * fd;
			slice_integ += fd;
			f2bd += f2d;
		}

		d[(j + 1) * MXDIM1 + i] = f2bd;


		// Calculations below are for the variance estimate of the integral estimate within the slice
		// math gives = 1/((npb)^2(npb-1) * N^2)[npb*f2bp - fbp^2]

		f2bd *= npb;
		f2bd = sqrt(f2bd);
		f2bd = (f2bd - slice_integ) * (f2bd + slice_integ) ; //accomplishes the math,
		//realden is N^2(npb)^2(npb-1)
		if (f2bd <= TINY)
			f2bd = TINY;


		slice_integ = blockReduceSum(slice_integ);
		f2bd = blockReduceSum(f2bd);


		if (tx == 0) {
			atomicAdd(&result_dev[0], f2bd);
			atomicAdd(&result_dev[1], slice_integ);
		}


		//result_dev[0] += f2bd;
		// }
	}

}

void adjustment(double xi[], double d[], double r[], double xin[], int N,
                int N1, int ndim) {
	double rc, xn, xo;
	int i, j;
	//int tx = threadIdx.x;

	double dt[MXDIM + 1];
	//j = blockIdx.x * blockDim.x + threadIdx.x;
	//if(j<ndim){

	//printarray(d, MXDIM1*N1);
	for (j = 1; j <= ndim; j++) {
		xo = d[1 * MXDIM1 + j];
		xn = d[2 * MXDIM1 + j];
		d[1 * MXDIM1 + j] = (xo + xn) / 2.0;
		dt[j] = d[1 * MXDIM1 + j];
		for (i = 2; i < N; i++) {
			rc = xo + xn;
			xo = xn;
			xn = d[(i + 1) * MXDIM1 + j];
			//printf("%d   %d   %e    %e\n", j, i, xo, xn);
			d[i * MXDIM1 + j] = (rc + xn) / 3.0;
			//printf("%d   %d    %e\n", j, i, d[i * MXDIM1 + j]);
			dt[j] += d[i * MXDIM1 + j];
		}
		d[N * MXDIM1 + j] = (xo + xn) / 2.0;
		//printf("outi  %d   %e   %e\n", j, xo, xn);
		dt[j] += d[N * MXDIM1 + j];

	}

	for (j = 1; j <= ndim; j++) {
		rc = 0.0;
		for (i = 1; i <= N; i++) {
			if (d[i * MXDIM1 + j] < TINY) d[i * MXDIM1 + j] = TINY;
			r[i] = pow((1.0 - d[i * MXDIM1 + j] / dt[j]) /
			           (log(dt[j]) - log(d[i * MXDIM1 + j])), ALPH);
			//printf("i , r,  %d    %e\n", i , r[i]);
			rc += r[i];
		}
		//printarray(r, N1);
		rebin(rc / N, N, r, xin, &xi[j * N1]);
	}

}

template <typename IntegT, int NDIM>
int VegasGPU (IntegT integrand) {
	
	IntegT* d_integrand;
    cudaMallocManaged((void**)&d_integrand, sizeof(IntegT));
	memcpy(d_integrand, &integrand, sizeof(IntegT));
	
	int i, j, iter;
	int npb;
	int ndim;

	double tsid, dim_integ;
	double rel_err = 0.0;
	float exp_rel_err;
	

	double wgt;
	double uwidth;



	int nblocks;

//double minNumEval = 43008;   //= 84 * 128 * 4  , num of streaming processors on v100 = 84
	int N, N1;



	printf("Enter the expected relative error:\n");
	scanf("%e", &exp_rel_err);
	//printf("the expected relative error %e\n", exp_rel_err);

//printf("Enter the number of dimensions:\n");
//scanf("%d", &ndim);


	int nthreads = 32;
	N = 8000;
	npb = 16000; 
	ndim = 9;
	double numEval = double(N) * npb * ndim ;
	printf("the num of eval is %f\n", numEval );
	N1 = N + 1;
	nblocks = int( ((N * ndim) + nthreads - 1) / nthreads);
	double totalSliceNum = ndim * N;

	double d[(N + 1) * (MXDIM + 1)];
	double r[N + 1];
	double xin[N + 1];
	double xi[(MXDIM + 1) * (N + 1)];



	double regn[2 * MXDIM + 1] = {0};
	double dx[MXDIM + 1] = {0};
	double result[2] = {0};
	double xjacd;
	double tmp = 0.0;

	double *xi_dev, *d_dev, *result_dev, *dt_dev, *r_dev, *xin_dev, *integralCal_dev, *regn_dev, *dx_dev;
	cudaMalloc((void**)&d_dev, sizeof(double) * (N + 1) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&xi_dev, sizeof(double) * (MXDIM + 1) * (N + 1)); cudaCheckError();
	cudaMalloc((void**)&result_dev, sizeof(double) * 2); cudaCheckError();
	cudaMalloc((void**)&regn_dev, sizeof(double) * (2 * MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&dx_dev, sizeof(double) * (MXDIM + 1)); cudaCheckError();

	cudaMalloc((void**)&dt_dev, sizeof(double) * (MXDIM + 1)); cudaCheckError();
	cudaMalloc((void**)&r_dev, sizeof(double) * (N + 1)); cudaCheckError();
	cudaMalloc((void**)&xin_dev, sizeof(double) * (N + 1)); cudaCheckError();
	cudaMalloc((void**)&integralCal_dev, sizeof(double) * 6); cudaCheckError();


	/* initialize withuniform grid spacing */
	uwidth = 1.0 / N;
	for (i = 1; i <= ndim; i++) {
		regn[i] = -1.0;
	}
	for (i = ndim + 1; i <= 2 * ndim; i++) {
		regn[i] = 1.0;
	}
	cudaMemcpy(regn_dev, regn, sizeof(double) * (2 * MXDIM + 1), cudaMemcpyHostToDevice); cudaCheckError();

	for (i = 0; i < MXDIM1 * N1 - 1; i++) {
		xi[i] = 0.0;
	}

	for (i = 1; i <= ndim; i++) {
		for (j = 0; j <= N; j++) {
			xi[i * N1 + j] = j * uwidth;
			//printf("xi j is %f", xi[i*N1 + j]);
		}
	}




	double si, swgt, schi, intgral, chi2a, sd;
	si = 0.0;  swgt = 0.0; schi = 0.0;

	xjacd = 1.0 / (N * npb * ndim);
	for (j = 1; j <= ndim; j++) {
		tmp = regn[j + ndim] - regn[j];
		dx[j] = tmp;
		xjacd *= tmp;
	}
//printf("xjacd is %e\n", xjacd);
	cudaMemcpy(dx_dev, dx, sizeof(double) * (MXDIM + 1), cudaMemcpyHostToDevice); cudaCheckError();
	printf("Total num of slices are: %f\n", totalSliceNum);
	

	printf("it, intgral, sd, chi2a, rel_err\n");
	for (iter = 1; iter <= ITMAX; iter++) {
		cudaMemcpy( xi_dev, xi, sizeof(double) * (MXDIM + 1) * (N + 1), cudaMemcpyHostToDevice) ; cudaCheckError();
		cudaMemset(d_dev, 0, sizeof(double) * (N + 1) * (MXDIM + 1));
		cudaMemset(result_dev, 0, 2 * sizeof(double)); //result[0]=tsid , result[1]=dim_integ

		vegas <<< nblocks, nthreads>>>(iter, dx_dev, xjacd , regn_dev , result_dev,
		                               totalSliceNum, npb, xi_dev, d_dev, N, N1, ndim, iter);

		cudaDeviceSynchronize ();

		cudaMemcpy(xi, xi_dev, sizeof(double) * (MXDIM + 1) * (N + 1), cudaMemcpyDeviceToHost); cudaCheckError();
		cudaMemcpy( d, d_dev, sizeof(double) * (N + 1) * (MXDIM + 1), cudaMemcpyDeviceToHost) ; cudaCheckError();
		cudaMemcpy(result, result_dev, sizeof(double) * 2, cudaMemcpyDeviceToHost);


		tsid = result[0];
		//printf("tsid in iteration is %d %e\n", iter, tsid);
		dim_integ = result[1];
		tsid *= 1.0 / (npb - 1.0);
		//itcounter = iter - 1;
	        //printf("dim_integ and variance : %d    %e     %e\n", iter, dim_integ, tsid);

		if (iter > SKIP) {
			wgt = 1.0 / tsid;
			si += wgt * dim_integ;
			schi += wgt * dim_integ * dim_integ;
			swgt += wgt;
			intgral = si / swgt;
			chi2a = (schi - si * (intgral)) / (iter - 0.9999);
			sd = sqrt(1.0 / swgt);
			rel_err = abs(sd / intgral);
			printf("%d, %.15f, %.15f, %.8f, %.15f,\n", iter, intgral, sd, chi2a, rel_err );
			//if(sd < exp_rel_err) break;
		}

		adjustment(xi, d, r, xin, N, N1, ndim);

	}


	cudaFree(d_dev);
	cudaFree(xi_dev);
	cudaFree(xin_dev);
	cudaFree(r_dev);
	cudaFree(dt_dev);
	cudaFree(result_dev);
	cudaFree(integralCal_dev);

}