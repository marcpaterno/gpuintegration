//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"



class GENZ_2_8D {
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
	const double a = 50.;
    const double b = .5;
    const double term_1 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(x - b, 2.));
    const double term_2 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(y - b, 2.));
    const double term_3 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(z - b, 2.));
    const double term_4 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(w - b, 2.));
    const double term_5 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(v - b, 2.));
    const double term_6 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(u - b, 2.));
	const double term_7 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(t - b, 2.));
	const double term_8 = 1. / ((1. / sycl::pow(a, 2.)) + sycl::pow(s - b, 2.));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6 * term_7 * term_8;
    return val;
  }
};

class GENZ_3_8D{
public:
  SYCL_EXTERNAL double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
	return sycl::pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
  }
};

class GENZ_4_8D {
  public:
    SYCL_EXTERNAL double
    operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
    {
	  double beta = .5;
      return sycl::exp(
        -1.0 * (sycl::pow(25., 2.) * sycl::pow(x - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(y - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(z - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(w - beta, 2.) +
                sycl::pow(25., 2.) * sycl::pow(v - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(u - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(t - beta, 2.) + 
				sycl::pow(25., 2.) * sycl::pow(s - beta, 2.)));
    }
};

class GENZ_5_8D {
public:
  SYCL_EXTERNAL double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
	double beta = .5;
    double t1 = -10. * sycl::fabs(x - beta) - 10. * sycl::fabs(y - beta) -
                10. * sycl::fabs(z - beta) - 10. * sycl::fabs(k - beta) -
                10. * sycl::fabs(m - beta) - 10. * sycl::fabs(n - beta) -
                10. * sycl::fabs(p - beta) - 10. * sycl::fabs(q - beta);
    return sycl::exp(t1);
  }
};

class GENZ_6_8D {
public:
  SYCL_EXTERNAL double
  operator()(double u, double v, double w, double x, double y, double z, double p, double t)
  {
	if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4 || p > .3 || t > .2)
      return 0.;
    else
      return sycl::exp(10. * z + 9. * y + 8. * x + 7. * w + 6. * v + 5. * u + 4. *p + 3. * t);
  }
};


template<typename F>
double execute_integrand(std::array<double, 8> point){
	const size_t num_blocks = 1024;
	const size_t num_threads = 64;
	const size_t num_invocations = 10000;
	
	F integrand;  
	F* d_integrand = quad::cuda_copy_to_managed(integrand);
	
	double* d_point = cuda_malloc<double>(point.size());
	cuda_memcpy_to_device(d_point, point.data(), point.size());
	
	double* output = cuda_malloc<double>(num_threads*num_blocks);
	
	//kernel<F><<<num_blocks, num_threads>>>(d_integrand, d_point, output, num_invocations);
	//cudaDeviceSynchronize();
	auto q = 	sycl::queue(sycl::gpu_selector()/*, sycl::property::queue::enable_profiling{}*/);
	
	for(int i = 0; i < 10; ++i)
    /*sycl::event e = */q.submit([&](sycl::handler& cgh) {
	
	
		cgh.parallel_for(
			   sycl::nd_range(sycl::range(num_blocks*num_threads) , sycl::range(num_threads)),
			   [=](sycl::nd_item<1> item_ct1)
			   [[intel::reqd_sub_group_size(32)]] {
			   		   
			size_t tid = item_ct1.get_local_id(0) + item_ct1.get_local_range().get(0) * item_ct1.get_group(0);
			double total = 0.;
			gpu::cudaArray<double, 8> point;
	
			for(int i=0; i < 8; ++i){
				point[i] = d_point[i];
			}
	
			for(int i=0; i < num_invocations; ++i){
		
				//double res = point[0] / point[1] / point[2] / point[3] / point[4] / point[5] / point[6] / point[7];
				//double res = d_integrand->operator()(point[0], point[1], point[2], point[3], point[4], point[5], point[6], point[7]);
				double res = gpu::apply(*d_integrand, point);
				total += res;
			}
			output[tid] = total;
			     
		});
    }).wait();
	
		
	std::vector<double> host_output;
	host_output.resize(num_threads*num_blocks);
	//std::cout<<"vector size:"<<host_output.size()<<std::endl;
	cuda_memcpy_to_host<double>(host_output.data(), output, host_output.size());
	
	double sum = 0.;
	for(int i=0; i < num_threads*num_blocks; ++i)
		sum += host_output[i];
	
	
	
	sycl::free(output, q);
	sycl::free(d_integrand, q);
	sycl::free(d_point, q);
	return sum;
}


int main(){
  std::array<double, 8> point = {0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8};
  double sum = 0.;
  sum += execute_integrand<GENZ_2_8D>(point);
  sum += execute_integrand<GENZ_3_8D>(point);
  sum += execute_integrand<GENZ_4_8D>(point);
  sum += execute_integrand<GENZ_5_8D>(point);
  sum += execute_integrand<GENZ_6_8D>(point);
  printf("%.15e\n", sum);
  
    return 0;
}

