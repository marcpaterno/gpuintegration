#ifndef ONE_API_FINISHED_ESTIMATES_H
#define ONE_API_FINISHED_ESTIMATES_H
#include <CL/sycl.hpp>
#include "oneAPI/quad/util/cuhreResult.h"
#include "oneAPI/quad/Region_characteristics.h"
#include "oneAPI/quad/Region_estimates.h"
//#include "oneapi/mkl.hpp"
#include "oneAPI/quad/util/MemoryUtil.h"
#include <iostream>

template<size_t ndim>
cuhreResult<double>
compute_finished_estimates(sycl::queue& q, const Region_estimates<ndim>& estimates, const Region_characteristics<ndim>& classifiers, const cuhreResult<double>& iter){
    
    using namespace sycl;
    
    cuhreResult<double> finished;
    size_t stride = 1;
    size_t n = estimates.size;
    
	//CANNOT USE mkl::blas::column_major?
	
    double* dp_res = malloc_device<double>(1, q);
	double* h_res = new double;
	
    event est_ev = oneapi::mkl::blas::column_major::dot(q, n, estimates.integral_estimates, stride, classifiers.active_regions, stride , dp_res);
    est_ev.wait();
	quad::copy_to_host<double>(h_res, dp_res, 1);
    finished.estimate = iter.estimate - h_res[0];

    event errorest_ev = oneapi::mkl::blas::column_major::dot(q, n, estimates.error_estimates, stride, classifiers.active_regions, stride , dp_res);
    errorest_ev.wait();
	quad::copy_to_host<double>(h_res, dp_res, 1);
    finished.errorest = iter.errorest - h_res[0];    
	delete h_res;
	sycl::free(dp_res, q);
    return finished;
}

#endif
