#include "cuba.h"
#include "cubacpp/cubacpp.hh"
#include "cubacpp/vegas.hh"
#include "cubacpp/integration_result.hh"
#include <iostream>
#include <chrono>
#include <string>
#include <cmath>
#define PI 3.14159265358979323844

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;


double sinSum6D (double x, double y, double z, double k, double l, double m)
{
    return sin(x + y + z + k + l + m);
}

double Gauss9D(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o,
             double p)
  {
    double sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
                 pow(m, 2) + pow(n, 2) + pow(o, 2) + pow(p, 2);
    return exp(-1 * sum / (2 * pow(0.01, 2))) *
           (1 / pow(sqrt(2 * PI) * 0.01, 9));
  }

double Genz1_8D(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
{
    return cos(s + 2. * t + 3. * u + 4. * v + 5. * w + 6. * x + 7. * y +
               8. * z);
}
  
  
double Genz2_6D(double x, double y, double z, double k, double l, double m)
{
    double a = 50.;
    double b = .5;

    double term_1 = 1. / ((1. / pow(a, 2)) + pow(x - b, 2));
    double term_2 = 1. / ((1. / pow(a, 2)) + pow(y - b, 2));
    double term_3 = 1. / ((1. / pow(a, 2)) + pow(z - b, 2));
    double term_4 = 1. / ((1. / pow(a, 2)) + pow(k - b, 2));
    double term_5 = 1. / ((1. / pow(a, 2)) + pow(l - b, 2));
    double term_6 = 1. / ((1. / pow(a, 2)) + pow(m - b, 2));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val /*/((1.286889807581113e+13))*/;
}
  
double Genz3_3D(double x, double y, double z)
{
      return pow(1 + 3 * x + 2 * y + z, -4);
}
 
 
 double Genz3_8D(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
{
      return pow(1 + 8 * s + 7 * t + 6 * u + 5 * v + 4 * w + 3 * x + 2 * y + z,
                 -9);
}

double Genz4_5D(double x, double y, double z, double w, double v)
{
      // double alpha = 25.;
      double beta = .5;
      return exp(
        -1.0 * (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2)));
}

double Genz4_8D(double x,
               double y,
               double z,
               double w,
               double v,
               double k,
               double m,
               double n)
{
      // double alpha = 25.;
      double beta = .5;
      return exp(
        -1.0 * (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2) + pow(25, 2) * pow(k - beta, 2) +
                pow(25, 2) * pow(m - beta, 2) + pow(25, 2) * pow(n - beta, 2)));
}

double Genz5_8D(double x,
               double y,
               double z,
               double k,
               double m,
               double n,
               double p,
               double q)
{
      double beta = .5;
      double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                  10. * fabs(z - beta) - 10. * fabs(k - beta) -
                  10. * fabs(m - beta) - 10. * fabs(n - beta) -
                  10. * fabs(p - beta) - 10. * fabs(q - beta);
      return exp(t1);
}

double Genz6_6D(double u, double v, double w, double x, double y, double z)
{
      if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
        return 0.;
      else
        return exp(10 * z + 9 * y + 8 * x + 7 * w + 6 * v +
                   5 * u) /*/1.5477367885091207413e8*/;
}

template<typename F>
bool time_and_call(F f, std::string integrandID, double epsrel, double correct_answer, cubacpp::IntegrationVolume<cubacpp::arity<F>()> vol){
     cubacpp::Vegas vegas;
     double epsabs = 1.0e-12;
     vegas.maxeval =  2e9;
     
     auto t0 = std::chrono::high_resolution_clock::now();
     auto res = vegas.integrate (f, epsrel, epsabs, vol);
     MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0; 

     //if(res.status == 0)
        std::cout<<integrandID<<","
		    <<std::scientific<<correct_answer<<","
			<<epsrel<<","
			<<epsabs<<","
			<<std::scientific<<res.value<<","
			<<std::scientific<<res.error<<","
			<<res.status<<","
			<<dt.count()<<std::endl;
    //  return res.status;
    return 0;
}

/*void Exec_SinSum6D(){
    constexpr int ndim = 6;
    double true_value = -49.165073;
    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 10., 10., 10., 10., 10., 10.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    
    cubacpp::Vegas vegas;
    
    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    
    while(epsrel >= 1.0e-9)
    {
        unsigned long long calls = 1000 * 1000 * 1000;

        vegas.maxeval = calls;
        double error = 0., value = 0.;
        do
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = vegas.integrate (&sinSum6D, epsrel, epsabs, vol);
            MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0; 
            
            error = res.error;
            value = res.value;
            
            if(fabs( res.error/res.value) <= epsrel)
                display_results ("SinSum6D", res.value,  res.error, epsrel, true_value, calls, dt.count());
            else{
                calls *= 10;
            }
        }
        while(fabs( error/value) > epsrel && calls < 1e9);
                
            if(fabs(error/value) > epsrel)
                break;
        epsrel /= 5;
    }
}*/


void Exec_SinSum6D(){
    constexpr int ndim = 6;
    double true_value = -49.165073;
    double epsrel_min = 1e-8;

    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 10., 10., 10., 10., 10., 10.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(sinSum6D, "SinSum6D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Gauss9D(){
    constexpr int ndim = 9;
    double true_value = 1.;
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { -1., -1., -1., -1., -1., -1., -1., -1., -1.};
    cubacpp::array<ndim> highs = { 1., 1., 1., 1., 1., 1., 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Gauss9D, "Gauss9D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Genz1_8D(){
    constexpr int ndim = 8;
    double true_value =
    (1. / 315.) * sin(1.) * sin(3. / 2.) * sin(2.) * sin(5. / 2.) * sin(3.) *
    sin(7. / 2.) * sin(4.) *
    (sin(37. / 2.) - sin(35. / 2.)); 
    
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 1., 1., 1., 1., 1., 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Genz1_8D, "f1 8D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Genz2_6D(){
    constexpr int ndim = 6;
    double true_value = 1.286889807581113e+13;  
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 1., 1., 1., 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Genz2_6D, "f2 6D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Genz3_3D(){
    constexpr int ndim = 3;
    double true_value = 0.010846560846560846561;
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { 0., 0., 0.};
    cubacpp::array<ndim> highs = { 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Genz3_3D, "f3 3D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Genz3_8D(){
    constexpr int ndim = 8;
    double true_value = 2.2751965817917756076e-10;
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 1., 1., 1., 1., 1., 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Genz3_8D, "f3 8D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Genz4_5D(){
    constexpr int ndim = 5;
    double true_value = 1.79132603674879e-06;
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 1., 1., 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Genz4_5D, "f4 5D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Genz4_8D(){
    constexpr int ndim = 8;
    double true_value = (6.383802190004379e-10);
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 1., 1., 1., 1., 1., 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Genz4_8D, "f4 8D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Genz5_8D(){
    constexpr int ndim = 8;
    double true_value = 2.425217625641885e-06;
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 1., 1., 1., 1., 1., 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Genz5_8D, "f5 8D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

void Exec_Genz6_6D(){
    constexpr int ndim = 6;
    double true_value = 1.5477367885091207413e8;
    double epsrel_min = 1e-8;
    cubacpp::array<ndim> lows  = { 0., 0., 0., 0., 0., 0.};
    cubacpp::array<ndim> highs = { 1., 1., 1., 1., 1., 1.};
    cubacpp::IntegrationVolume<ndim> vol(lows, highs);
    
    bool success = 0;
    double epsrel = 1.e-3;
    
    while(time_and_call(Genz6_6D, "f6 6D", epsrel, true_value, vol) == success){
        epsrel /= 5.;
        if(epsrel < epsrel_min)
            break;
    }
}

int
main (void)
{
  //Exec_SinSum6D();
  //Exec_Gauss9D();
  ///Exec_Genz1_8D();
  //Exec_Genz2_6D();
  //Exec_Genz3_3D();
	Exec_Genz3_8D();
  //Exec_Genz4_5D();
  Exec_Genz4_8D();
  //Exec_Genz5_8D();
  //Exec_Genz6_6D();
  
  return 0;
}
