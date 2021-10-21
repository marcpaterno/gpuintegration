#include <math.h>
// define a new function and update function specification at the end 

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


__inline__ __device__  double func3(double *xx, int ndim) {
	double k = 0.01890022674239546529975841;
	double t1 = 4.0*k*xx[1]*xx[1];
	double t2 = (0.01 + (xx[1]-xx[2]-0.33333333333333333)*(xx[1]-xx[2]-0.33333333333333333));
	return t1/t2;
}

//fcode 3
__inline__ __device__
double BoxIntegral8_22(double *x, double dim){
	double s = 22;
    double sum = 0;
    sum = pow(x[1], 2) + pow(x[2], 2) + pow(x[3], 2) + pow(x[4], 2) + pow(x[5], 2) +
          pow(x[6], 2) + pow(x[7], 2) + pow(x[8], 2);
    return pow(sum, s / 2);
}

//fcode 4
__inline__ __device__
double GENZ1_8D(double *x, double dim){
   return cos(x[1] + 2. * x[2] + 3. * x[3] + 4. * x[4] + 5. * x[5] + 6. * x[6] + 7. * x[7] + 8. * x[8]); 

}

//fcode 5
__inline__ __device__
double GENZ2_2D(double *x, double dim){
    double a = 50.;
    double b = .5;

    double term_1 = 1./((1./pow(a,2)) + pow(x[1]- b, 2));
    double term_2 = 1./((1./pow(a,2)) + pow(x[2]- b, 2));

    double val  = term_1 * term_2;
    return val;
}

//fcode 6
__inline__ __device__
double GENZ2_6D(double *x, double dim){
    double a = 50.;
    double b = .5;

    double term_1 = 1./((1./pow(a,2)) + pow(x[1]- b, 2));
    double term_2 = 1./((1./pow(a,2)) + pow(x[2]- b, 2));
	double term_3 = 1./((1./pow(a,2)) + pow(x[3]- b, 2));
	double term_4 = 1./((1./pow(a,2)) + pow(x[4]- b, 2));
	double term_5 = 1./((1./pow(a,2)) + pow(x[5]- b, 2));
	double term_6 = 1./((1./pow(a,2)) + pow(x[6]- b, 2));
	
    double val  = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
	return val;
}

//fcode 7
__inline__ __device__
double GENZ3_3D(double *x, double dim){
  
    return pow(1+3*x[1]+2*x[2]+x[3], -4);
}

//fcode 8
__inline__ __device__
double GENZ4_5D(double *x, double dim){
   
    double beta = .5;
    return exp(-1.0*(pow(25,2)*pow(x[1]-beta, 2) + 
				         pow(25,2)*pow(x[2]-beta, 2) +
				         pow(25,2)*pow(x[3]-beta, 2) +
				         pow(25,2)*pow(x[4]-beta, 2) +
				         pow(25,2)*pow(x[5]-beta, 2))
				  );
}

//fcode 9
__inline__ __device__
double GENZ5_8D(double *x, double dim){
	    double beta = .5;
        double t1 = -10.*fabs(x[1] - beta) - 10.* fabs(x[2] - beta) - 10.* fabs(x[3] - beta) - 10.* fabs(x[4] - beta) - 10.* fabs(x[5] - beta)-10.*fabs(x[6] - beta)-10.*fabs(x[7] - beta)-10.*fabs(x[8] - beta);
        return exp(t1);
}

//fcode 10
__inline__ __device__
double BoxIntegral8_15(double *x, double dim){
    double s = 15;
    double sum = 0;
    sum = pow(x[1], 2) + pow(x[2], 2) + pow(x[3], 2) + pow(x[4], 2) + pow(x[5], 2) +
          pow(x[6], 2) + pow(x[7], 2) + pow(x[8], 2);
    return pow(sum, s / 2);
}

//fcode 11
__inline__ __device__
double GENZ6_6D(double *x, double dim){
    if(x[6] > .9 || x[5] > .8 || x[4] > .7 || x[3] > .6 || x[2] >.5 || x[1] > .4)
        return 0.;
    else
        return exp(10*x[6] + 9*x[5] + 8*x[4] + 7*x[3] + 6*x[2] + 5*x[1]);

}

//fcode 12
__inline__ __device__
double GENZ4_8D(double *x, double dim){
   
    double beta = .5;
    return exp(-1.0*(pow(25,2)*pow(x[1]-beta, 2) + 
				         pow(25,2)*pow(x[2]-beta, 2) +
				         pow(25,2)*pow(x[3]-beta, 2) +
				         pow(25,2)*pow(x[4]-beta, 2) +
				         pow(25,2)*pow(x[5]-beta, 2) + 
                         pow(25,2)*pow(x[6]-beta, 2) +
                         pow(25,2)*pow(x[7]-beta, 2) +
                         pow(25,2)*pow(x[8]-beta, 2))
				  );
}

//fcode 13
__inline__ __device__
double GENZ3_8D(double *x, double dim){
   
    return pow(1 + 8 * x[8] + 7 * x[7] + 6 * x[6] + 5 * x[5] + 4 * x[4] + 3 * x[1] + 2 * x[2] + x[3],
                 -9);
}

__inline__ __device__
double roosarnoldthree(double* rx, int dim)
{
	double value = 1./sqrtf(powf(M_PI*M_PI/8., (double)dim)-1.);
	for (int i = 1; i <= dim; i++){
		value *= (M_PI/2.*sinf(M_PI*rx[i])-1.);
	}
	return value;
}

__inline__ __device__
double rst(double* rx, int dim)
{
	double value = 1./sqrtf(powf(1.+1./12.,(double)dim)-1.);
	for (int i = 1; i <= dim; i++){
		value *= ((fabsf(4.*rx[i]-2.)+1.)/2.-1.);
	}
	return value;
}

//this function depends on index, can't swap between start at zero and one
__inline__ __device__
double sobolprod(double* rx, int dim)
{
	double value = 1.;
	for (int i = 1; i <= dim; i++){
		value *= (1.+1./((double)(3*(i+2)*(i+2))));
	}
	value = sqrtf(1./(value-1.));
	for (int i = 1; i <= dim; i++){
		value *= ((double)(i+1)+2.*rx[i])/(double)(i + 2)-1.;
	}
	return value;
}

__inline__ __device__
double oscill(double* rx, int dim)
{
	double value = 2.*M_PI;
	double p = 1.;
	for (int i = 1; i <= dim; i++){
		value += rx[i];
		p *= sinf(0.5);
	}
	value = cos(value)-pow(2., (double)dim)*cosf(2.*M_PI+0.5*(double)dim)*p;
	return value;
}

__inline__ __device__
//Choosing beta_i = 0.5 and alpha_i = 1 for every i.
double prpeak(double* rx, int dim)
{
	double value = 1.;
	double e = 1.;
	for (int i = 1; i <= dim; i++){
		value *= 1./(1+(rx[i]-0.5)*(rx[i]-0.5));
		e *= (atan(0.5f)-atan(-0.5));
	}
	value += -e;
	return value;
}

__inline__ __device__

double sum(double* rx, int dim)
{
	double value = 0.;
	for (int i = 1; i <= dim; i++){
		value += rx[i];
	}
	value = 1. / sqrt((double)dim/12.) * (value - (double)dim / 2.);
	return value;
}

__inline__ __device__
double sqsum(double* rx, int dim)
{
	double value = 0.;
	for (int i = 1; i <= dim; i++){
		value += rx[i] * rx[i];
	}
	value = sqrt(45. / (4. * (double)dim)) * (value - (double)dim / 3);
	return value;
}

__device__
float sqsumFloat(float* rx, int dim)
{
	float value = 0.f;
	for (int i = 1; i <= dim; i++){
		value += rx[i] * rx[i];
	}
	value = sqrtf(45.f / (4.f * (float)dim)) * (value - (float)dim / 3);
	return value;
}

__inline__ __device__
double sumsqroot(double* rx, int dim)
{
	double value = 0.;
	for (int i = 1; i <= dim; i++){
		value += sqrt(rx[i]);
	}
	value = sqrt(18. / (double)dim) * (value - 2./3. * (double)dim);
	return value;
}

__inline__ __device__

double prodones(double* rx, int dim)
{
	double value = 1.;
	for (int i = 1; i <= dim; i++){
		value *= copysign(1., rx[i]-0.5);
	}
	return value;
}

__inline__ __device__

double prodexp(double* rx, int dim)
{
	double e = sqrt((15. * exp(15.) + 15.) / (13. * exp(15.) + 17.));
	e = pow(e, double(dim) * 0.5);
	double value = 1.;
	for (int i = 1; i <= dim; i++){
		value *= ((exp(30. * rx[i] - 15.)) - 1.) / (exp(30. * rx[i] - 15.) + 1.);		
	}
	value *= e;
	return value;
}

__inline__ __device__

double prodcub(double* rx, int dim)
{
	double value = 1.;
	for (int i = 1; i <= dim; i++){
		value *= (-2.4f*sqrt(7.)*(rx[i]-0.5f)+8.f*sqrt(7.f)*(rx[i]-0.5f)*(rx[i]-0.5f)*(rx[i]-0.5f));
	}
	return value;
}

__inline__ __device__

//PRODX has a lot of extremes when dimensions are big, it's expected to not do well
double prodx(double* rx, int dim)
{
	double value = 1.f;
	for (int i = 1; i <= dim; i++){
		value *= (rx[i] - 0.5f);
	}
	value *= pow(2.f*sqrt(3.f), (double) dim);
	return value;
}

//this function may be problematic id 20
__inline__ __device__
double sumfifj(double* rx, int dim)
{
	double value = 0.f;
	for (int i = 1; i <= dim; i++){
		double aux = 0.f;
		for (int j = 1; j < i; j++){
			aux += copysign(1.,(1./6.-rx[j])*(rx[j]-4./6.));
		}
		value += copysign(1.,(1./6.-rx[i])*(rx[i]-4./6.))*aux;
	}
	value *= sqrt(2./(double)(dim*(dim-1)));
	return value;
}

__inline__ __device__
double sumfonefj(double* rx, int dim)
{
	double value = 0.f;
	for (int i = 1; i < dim; i++){
		value += 27.20917094*rx[i]*rx[i]*rx[i]-36.1925085*rx[i]*rx[i]+8.983337562*rx[i]+0.7702079855;
	}
	value *= (27.20917094*rx[0]*rx[0]*rx[0]-36.1925085*rx[0]*rx[0]+8.983337562*rx[0]+0.7702079855)/sqrt((double)dim-1.f);
	return value;
}

__inline__ __device__

double hellekalek(double* rx, int dim)
{
	double value = 1.;
	for (int i = 1; i < dim; i++){
		value *= ((rx[i] - 0.5f)/sqrt(12.));
	}
	return value;
}

__inline__ __device__

double roosarnoldone(double* rx, int dim)
{
	double value = 1./(double)dim;
	double aux = 0.;
	for (int i = 1; i < dim; i++){
		aux += abs(4.f*rx[i]-2.f)-1.f;
	}
	value *= aux;
	return value;
}

__inline__ __device__

//Can give huge error
double roosarnoldtwo(double* rx, int dim)
{
	double value = sqrt(1./(pow(4./3., (double)dim)-1.));
	for (int i = 1; i < dim; i++){
		value *= (abs(4.*rx[i]-2.) - 1.);
	}
	return value;
}

