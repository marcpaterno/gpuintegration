#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include "oneAPI/quad/Cubature_rules.h"
#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/GPUquad/Rule.h"
#include "oneAPI/quad/GPUquad/Phases.h"
#include "oneAPI/quad/Cubature_rules.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>
#include "oneAPI/quad/Workspace.h"

#include "oneAPI/demos/demo_utils.h"

#include "oneapi/mkl.hpp"
#include "oneapi/tbb.h"
#include <limits>


void this_works(){

    sycl::queue q;
    constexpr size_t ndim = 3;
    size_t num_elems = 77;
    
    quad::Rule<double> rule;
    rule.Init(ndim, num_elems, 0, 0);
    
    constexpr size_t block_size = 64;            
    size_t feval = get_feval<ndim>();
    constexpr size_t permutations_size = get_permutation_size<ndim>();
    constexpr size_t nsets = 9;
    constexpr size_t nrules = 5;
    
    int* _gpuGenPermVarStart = quad::copy_to_shared<int>(q, rule.cpuGenPermVarStart, feval + 1);
    int* _gpuGenPermGIndex = quad::copy_to_shared<int>(q, rule.cpuGenPermGIndex, feval);
    int* _gpuGenPos = quad::copy_to_shared<int>(q, rule.genPtr, permutations_size);
    double* _gpuG = quad::copy_to_shared<double>(q, rule.cpuG, ndim * nsets);
       
    double* _cRuleWt = quad::copy_to_shared<double> (q, rule.CPURuleWt, nrules * nsets);
    double* _GPUScale = quad::copy_to_shared<double> (q, rule.CPUScale, nrules * nsets);
    double* _GPUNorm = quad::copy_to_shared<double> (q, rule.CPUNorm, nrules * nsets);
    int* _gpuGenPermVarCount = quad::copy_to_shared<int> (q, rule.cpuGenPermVarCount, feval);
    
    double* _generators = malloc_shared<double>(feval * ndim, q);
    
    q.submit([&](auto &cgh) {
        
        cgh.parallel_for(sycl::range<1>(feval), [=](sycl::id<1> feval_id){
            
            size_t tid = feval_id[0];
            double g[ndim] = {};
                
            int posCnt = _gpuGenPermVarStart[tid + 1] - _gpuGenPermVarStart[tid];
            int gIndex = _gpuGenPermGIndex[tid];
            
            for (int posIter = 0; posIter < posCnt; ++posIter) {
              int pos = _gpuGenPos[_gpuGenPermVarStart[tid] + posIter];
              int absPos = abs(pos);
                
              if (pos == absPos) {
              
                int write_index = (absPos - 1);
                size_t r_index = gIndex * ndim + posIter;
                g[absPos - 1] = _gpuG[gIndex * ndim + posIter];
                  
              } else {  
               
                int write_index = (absPos - 1);
                int r_index = (gIndex * (int)ndim + posIter);
                
                g[write_index] = -_gpuG[r_index];
              }
            }
            
            //output results
            for (size_t dim = 0; dim < ndim; dim++) {
                _generators[feval * dim + tid] = g[dim];
            }
                        
         });  
    }).wait_and_throw();
      
      for(int i=0; i < feval*ndim; ++i)
          printf("generator %i: %f\n", i, _generators[i] );

}

void this_doesnt_work(){
    
    sycl::queue q;
    constexpr size_t ndim = 3;
    size_t num_elems = 77;
    
    quad::Rule<double> rule;
    rule.Init(ndim, num_elems, 0, 0);
    
    constexpr size_t block_size = 64;            
    size_t feval = get_feval<ndim>();
    constexpr size_t permutations_size = get_permutation_size<ndim>();
    constexpr size_t nsets = 9;
    constexpr size_t nrules = 5;
    
    int* _gpuGenPermVarStart = quad::copy_to_shared<int>(q, rule.cpuGenPermVarStart, feval + 1);
    int* _gpuGenPermGIndex = quad::copy_to_shared<int>(q, rule.cpuGenPermGIndex, feval);
    int* _gpuGenPos = quad::copy_to_shared<int>(q, rule.genPtr, permutations_size);
    double* _gpuG = quad::copy_to_shared<double>(q, rule.cpuG, ndim * nsets);
       
    double* _cRuleWt = quad::copy_to_shared<double> (q, rule.CPURuleWt, nrules * nsets);
    double* _GPUScale = quad::copy_to_shared<double> (q, rule.CPUScale, nrules * nsets);
    double* _GPUNorm = quad::copy_to_shared<double> (q, rule.CPUNorm, nrules * nsets);
    int* _gpuGenPermVarCount = quad::copy_to_shared<int> (q, rule.cpuGenPermVarCount, feval);
    
    double* _generators = malloc_shared<double>(feval * ndim, q);
    
    q.submit([&](auto &cgh) {
        //sycl::stream str(8192, 1024, cgh);
        //sycl::stream str(65536, 1024, cgh);
        cgh.parallel_for(sycl::range<1>(feval), [=](sycl::id<1> feval_id){
            
            size_t tid = feval_id[0];
            double g[ndim];
            
            //this is the difference that makes it not work
            for (int dim = 0; dim < 3; dim++) {
              g[dim] = 0.;
            }
              
            int posCnt = _gpuGenPermVarStart[tid + 1] - _gpuGenPermVarStart[tid];
            int gIndex = _gpuGenPermGIndex[tid];
            
            for (int posIter = 0; posIter < posCnt; ++posIter) {
              int pos = _gpuGenPos[_gpuGenPermVarStart[tid] + posIter];
              int absPos = abs(pos);
                
              if (pos == absPos) {
              
                int write_index = (absPos - 1);
                size_t r_index = gIndex * ndim + posIter;
                g[absPos - 1] = _gpuG[gIndex * ndim + posIter];
                //str <<"feval:" << tid << " dim:" << write_index << " g:" << g[0] << "," << g[1] << "," << g[2] << "\n";      
              } else {  
               
                int write_index = (absPos - 1);
                int r_index = (gIndex * (int)ndim + posIter);
                
                g[write_index] = -_gpuG[r_index];
                //str <<"feval:" << tid << " dim:" << write_index << " g:" << g[0] << "," << g[1] << "," << g[2] << "\n";      
              }
            }
            
            //output results
            for (size_t dim = 0; dim < ndim; dim++) {
                _generators[feval * dim + tid] = g[dim];
                

            }
                        
         });  
    }).wait_and_throw();
      
      for(int i=0; i < feval*ndim; ++i)
          printf("generator %i: %f\n", i, _generators[i] );

}
  
int main(){
  this_works();
  std::cout<<"--------\n";
  this_doesnt_work();  
  return 0;
}
