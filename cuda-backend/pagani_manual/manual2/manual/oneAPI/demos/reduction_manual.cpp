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


using namespace sycl;

template<typename T>
using shared = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>;  

namespace exper{
  template<typename T, int blockdim>
  double
  block_reduce(sycl::nd_item<1> item, T val, shared<T> fast_access_buffer, sycl::stream str)
  {
    const size_t work_group_id = item.get_group_linear_id();
    const size_t work_group_tid = item.get_local_id();  
    
    auto sg = item.get_sub_group();
         
    int sg_id = sg.get_group_id()[0];  
    int l_id = sg.get_local_id()[0];

    val = sycl::reduce_over_group(sg, val, sycl::plus<>()); //warp reduction
        
    
    item.barrier(sycl::access::fence_space::local_space); //consider global_and_local

    if(l_id == 0)
      fast_access_buffer[sg_id] = val;
    
    item.barrier(sycl::access::fence_space::local_space); //consider global_and_local

    val = work_group_tid < sg.get_group_range()[0] ? fast_access_buffer[work_group_tid] : 0.; //only warp 0 writes to val

    item.barrier(sycl::access::fence_space::local_space);       

    //posible optimization
    //if(sg_id == 0)
    {  
      val = sycl::reduce_over_group(sg, val, sycl::plus<>());   
    }
    return val;
  }     
     
  template<typename T, int blockdim>
  double
  multi_warp_block_reduce(sycl::nd_item<1> item, T val, shared<T> fast_access_buffer){
  
   const size_t work_group_id = item.get_group_linear_id();
    const size_t work_group_tid = item.get_local_id();  
    
    auto sg = item.get_sub_group();
         
    int sg_id = sg.get_group_id()[0];  
    int l_id = sg.get_local_id()[0];

    val = sycl::reduce_over_group(sg, val, sycl::plus<>()); //warp reduction
        
    
    item.barrier(sycl::access::fence_space::local_space); //consider global_and_local

    if(l_id == 0)
      fast_access_buffer[sg_id] = val;
    
    item.barrier(sycl::access::fence_space::local_space); //consider global_and_local

    val = work_group_tid < sg.get_group_range()[0] ? fast_access_buffer[work_group_tid] : 0.; //only warp 0 writes to val

    item.barrier(sycl::access::fence_space::local_space);       

    //posible optimization
    //if(sg_id == 0)
    {  
      val = sycl::reduce_over_group(sg, val, sycl::plus<>());   
    }
    return val;
  }  
}


void reduce_over_work_group(){
    sycl::queue q;  

    const size_t reduction_array_size = 64;  
    const size_t work_group_size = 64;
    constexpr size_t warp_size = 32;
    const size_t num_sub_groups = work_group_size/warp_size;
    
    double* input = sycl::malloc_shared<double>(reduction_array_size, q);  
    double* results = malloc_shared<double>(reduction_array_size/work_group_size, q);  
    
    for(int i = 0; i < reduction_array_size; ++i){
      input[i] = 1. + static_cast<double>(i);
    }
    
    q.submit([&](auto &cgh) {
    sycl::stream str(2048*8, 2048*4, cgh);
           
        cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {
                
            auto wg = item.get_group();
            size_t gl_tid = item.get_global_id();
            double val = reduce_over_group(wg, input[gl_tid], sycl::plus<>());
            
            if(gl_tid == 0)
                results[gl_tid] = val;
        });
    }).wait();  
    
    std::cout<<"reduce_over_workgroup must be 2080.0 results[0]:"<<results[0]<<std::endl;
}
 

template<int warp_size>
void single_block_manual_block_reduce(){
    sycl::queue q;  

    const size_t reduction_array_size = 64;  
    const size_t work_group_size = 64;
    //constexpr size_t warp_size = 32;
    const size_t num_sub_groups = work_group_size/warp_size;
    
    double* input = sycl::malloc_shared<double>(reduction_array_size, q);  
    double* results = malloc_shared<double>(reduction_array_size/work_group_size, q);  
    
    for(int i = 0; i < reduction_array_size; ++i){
      input[i] = 1. + static_cast<double>(i);
    }
    

    q.submit([&](auto &cgh) {
            sycl::stream str(2048*8, 2048*4, cgh);
            
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range(num_sub_groups), cgh);
           
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {
                
                auto sg = item.get_sub_group();
                auto wg = item.get_group();
                
                size_t gl_tid = item.get_global_id();
                
                size_t wg_id = item.get_group_linear_id();
                size_t wg_tid = item.get_local_id();
                
                size_t sg_id = sg.get_group_id()[0];  
                size_t sg_tid = sg.get_local_id()[0];
                
                double sum = exper::multi_warp_block_reduce<double, 64>(item, input[gl_tid], sdata);
                if(gl_tid == 0)
                    results[gl_tid] = sum;
            });
       }).wait();  
    
        std::cout<<"single_block_manual_block_reduce must be 2080.0 results[0]:"<<results[0]<<std::endl;
}

template<int warp_size>
void two_block_manual_block_reduce(){
    sycl::queue q;  
    const size_t reduction_array_size = 128;  
    const size_t work_group_size = 64;
    //constexpr size_t warp_size = 32;
    const size_t num_sub_groups = work_group_size/warp_size;
    
    double* input = sycl::malloc_shared<double>(reduction_array_size, q);  
    double* results = sycl::malloc_shared<double>(reduction_array_size/work_group_size, q);  
    
    for(int i = 0; i < reduction_array_size; ++i){
          input[i] = 1. + static_cast<double>(i);
    }
    
    q.submit([=](auto &cgh) {
            sycl::stream str(8192, 1024, cgh);
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(num_sub_groups, cgh);
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {

                auto sg = item.get_sub_group();
                auto wg = item.get_group();
                
                size_t gl_tid = item.get_global_id();
                
                size_t wg_id = item.get_group_linear_id();
                size_t wg_tid = item.get_local_id();
                
                size_t sg_id = sg.get_group_id()[0];  
                size_t sg_tid = sg.get_local_id()[0];
                
                double val = exper::block_reduce<double, 64>(item, input[gl_tid], sdata, str);
                item.barrier(sycl::access::fence_space::local_space);
                
                if(wg_tid == 0)
                    results[wg_id] = val;
            });
       }).wait();  
        
        //CHECK(results[0] == Approx(2080.));
        //CHECK(results[1] == Approx(6176.));
        
        std::cout<<"two_block_manual_block_reduce must be 2080.0 results[0]:"<<results[0]<<std::endl;
        std::cout<<"two_block_manual_block_reduce must be 6176.0 results[1]:"<<results[1]<<std::endl;
}

template<int warp_size>
void two_block_reduce_over_work_group(){
    sycl::queue q;  
    const size_t reduction_array_size = 128;  
    const size_t work_group_size = 64;
    const size_t num_sub_groups = work_group_size/warp_size;
    
    double* input = sycl::malloc_shared<double>(reduction_array_size, q);  
    double* results = sycl::malloc_shared<double>(reduction_array_size/work_group_size, q);  
    
    for(int i = 0; i < reduction_array_size; ++i){
          input[i] = 1. + static_cast<double>(i);
    }
    
    q.submit([=](auto &cgh) {
            sycl::stream str(8192, 1024, cgh);
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(num_sub_groups, cgh);
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {

                auto sg = item.get_sub_group();
                auto wg = item.get_group();
                
                size_t gl_tid = item.get_global_id();
                
                size_t wg_id = item.get_group_linear_id();
                size_t wg_tid = item.get_local_id();
                
                size_t sg_id = sg.get_group_id()[0];  
                size_t sg_tid = sg.get_local_id()[0];
                
                //double val = exper::block_reduce<double, 64>(item, input[gl_tid], sdata, str);
                double val = reduce_over_group(wg, input[gl_tid], sycl::plus<>());
                item.barrier(sycl::access::fence_space::local_space);
                
                /*str <<"wg_id " << wg_id
                    << "\twg tid " << wg_tid
                    << "\tgl_tid "<< gl_tid
                    << "\tsg_tid " << sg_tid
                    << "\tsg_id " << sg_id
                    << "\tval:" << val
                    //<< "\tinput " << input[gl_tid]
                    << sycl::endl;    */
                
                if(wg_tid == 0)
                    results[wg_id] = val;
            });
       }).wait();  
        
        //CHECK(results[0] == Approx(2080.));
        //CHECK(results[1] == Approx(6176.));
        
        std::cout<<"two_block_reduce_over_work_group must be 2080.0 results[0]:"<<results[0]<<std::endl;
        std::cout<<"two_block_reduce_over_work_group must be 6176.0 results[1]:"<<results[1]<<std::endl;
}

template<int warp_size>
void single_block_reduce_over_work_group(){
    sycl::queue q;  

    const size_t reduction_array_size = 64;  
    const size_t work_group_size = 64;
    //constexpr size_t warp_size = 32;
    const size_t num_sub_groups = work_group_size/warp_size;
    
    double* input = sycl::malloc_shared<double>(reduction_array_size, q);  
    double* results = malloc_shared<double>(reduction_array_size/work_group_size, q);  
    
    for(int i = 0; i < reduction_array_size; ++i){
      input[i] = 1. + static_cast<double>(i);
    }
    

    q.submit([&](auto &cgh) {
            sycl::stream str(2048*8, 2048*4, cgh);
            
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range(num_sub_groups), cgh);
           
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {
                
                auto sg = item.get_sub_group();
                auto wg = item.get_group();
                
                size_t gl_tid = item.get_global_id();
                
                size_t wg_id = item.get_group_linear_id();
                size_t wg_tid = item.get_local_id();
                
                size_t sg_id = sg.get_group_id()[0];  
                size_t sg_tid = sg.get_local_id()[0];
                
                double sum = reduce_over_group(wg, input[gl_tid], sycl::plus<>());
                if(gl_tid == 0)
                    results[gl_tid] = sum;
            });
       }).wait();  
    
        std::cout << "single_block_reduce_over_work_group must be 2080.0 results[0]:" << results[0] << std::endl;
}

template<int warp_size2>
void single_block(){
    sycl::queue q;  

    const size_t reduction_array_size = 64;  
    const size_t work_group_size = 64;
    constexpr size_t warp_size = 32;
    const size_t num_sub_groups = work_group_size/warp_size;
    
    double* input = sycl::malloc_shared<double>(reduction_array_size, q);  
    double* results = malloc_shared<double>(reduction_array_size/work_group_size, q);  
    
    for(int i = 0; i < reduction_array_size; ++i){
      input[i] = 1. + static_cast<double>(i);
    }
    

    q.submit([&](auto &cgh) {
            sycl::stream str(0, 0, cgh);
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range(num_sub_groups), cgh);
           
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(16)]] {
                
                auto sg = item.get_sub_group();
                auto wg = item.get_group();
                
                size_t gl_tid = item.get_global_id();
                
                size_t wg_id = item.get_group_linear_id();
                size_t wg_tid = item.get_local_id();
                
                size_t sg_id = sg.get_group_id()[0];  
                size_t sg_tid = sg.get_local_id()[0];
                
                double val = exper::block_reduce<double, 64>(item, input[gl_tid], sdata, str);
                
                if(gl_tid == 0)
                    results[gl_tid] = val;
            });
       }).wait();  
    
        std::cout<<"results[0]:"<<results[0]<<std::endl;
}

template<int warp_size2>
void two_block(){
    sycl::queue q;  
    const size_t reduction_array_size = 128;  
    const size_t work_group_size = 64;
    constexpr size_t warp_size = 32;
    const size_t num_sub_groups = work_group_size/warp_size;
    
    double* input = sycl::malloc_shared<double>(reduction_array_size, q);  
    double* results = sycl::malloc_shared<double>(reduction_array_size/work_group_size, q);  
    
    for(int i = 0; i < reduction_array_size; ++i){
          input[i] = 1. + static_cast<double>(i);
    }
    
    q.submit([=](auto &cgh) {
            sycl::stream str(0, 0, cgh);
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(num_sub_groups, cgh);
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(16)]] {

                auto sg = item.get_sub_group();
                auto wg = item.get_group();
                
                size_t gl_tid = item.get_global_id();
                
                size_t wg_id = item.get_group_linear_id();
                size_t wg_tid = item.get_local_id();
                
                size_t sg_id = sg.get_group_id()[0];  
                size_t sg_tid = sg.get_local_id()[0];
                
                double val = exper::block_reduce<double, 64>(item, input[gl_tid], sdata, str);
                item.barrier(sycl::access::fence_space::local_space);
                
                if(wg_tid == 0)
                    results[wg_id] = val;
            });
       }).wait();  
        
        //CHECK(results[0] == Approx(2080.));
        //CHECK(results[1] == Approx(6176.));
        
        std::cout<<"results[0]:"<<results[0]<<std::endl;
        std::cout<<"results[1]:"<<results[1]<<std::endl;
}

int main()
{
    sycl::queue q;
    Workspace<3> workspace(q);
    Sub_regions<3> regions(q, 2);
    
    single_block<16>();
    two_block<16>();
}
