#ifndef QUAD_UTIL_CUDA_CUSTOM_FUNCTIONS_CUH
#define QUAD_UTIL_CUDA_CUSTOM_FUNCTIONS_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include "oneAPI/pagani/quad/GPUquad/Sample.dp.hpp"
#include "common/oneAPI/cudaDebugUtil.h"
#include <cmath>

#include <algorithm>

/*
	require blocks to be equal to size
*/

template<typename T>
void
device_custom_reduce(T* arr, size_t size, T* out, sycl::nd_item<1> item_ct1,
                     T *shared){
    T sum = 0.;
    //reduce multiple elements per thread
        const int tid =
          item_ct1.get_group(0) * item_ct1.get_local_range().get(0) +
          item_ct1.get_local_id(0);
        const int total_num_threads =
          item_ct1.get_local_range().get(0) * item_ct1.get_group_range(0);

        for (size_t i = tid; i < size; i += total_num_threads) {
		sum += arr[i];
	}

        sum = quad::blockReduceSum(sum, item_ct1, shared);

        if (item_ct1.get_local_id(0) == 0){
	  out[item_ct1.get_group(0)] = sum;
	}
}

template <typename T>
T
custom_reduce(T* arr, size_t size) {
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  size_t num_threads = 512;
  size_t max_num_blocks = 1024;
  size_t num_blocks =
    std::min((size + num_threads - 1) / num_threads, max_num_blocks);
  T* out = quad::cuda_malloc<T>(num_blocks);
  
  q_ct1.submit([&](sycl::handler& cgh) {
      sycl::accessor<T,
		     1,
		     sycl::access_mode::read_write,
		     sycl::access::target::local>
	shared_acc_ct1(sycl::range(8), cgh);

      cgh.parallel_for(
		       sycl::nd_range(sycl::range(num_blocks) *
				      sycl::range(num_threads),
				      sycl::range(num_threads)),
		       [=](sycl::nd_item<1> item_ct1)
		       [[intel::reqd_sub_group_size(32)]] {
			 device_custom_reduce(
					      arr,
					      size,
					      out,
					      item_ct1,
					      (T*)shared_acc_ct1.get_pointer());
		       });
    }).wait();

  q_ct1.submit([&](sycl::handler& cgh) {
      sycl::accessor<T,
		     1,
		     sycl::access_mode::read_write,
		     sycl::access::target::local>
	shared_acc_ct1(sycl::range(8), cgh);

      cgh.parallel_for(
		       sycl::nd_range(sycl::range(1024),
				      sycl::range(1024)),
		       [=](sycl::nd_item<1> item_ct1)
		       [[intel::reqd_sub_group_size(32)]] {
			 device_custom_reduce(
					      out,
					      num_blocks,
					      out,
					      item_ct1,
					      (T*)shared_acc_ct1.get_pointer());
		       });
    }).wait();

  T res = 0;
  quad::cuda_memcpy_to_host<T>(&res, out, 1);
  sycl::free(out, q_ct1);
  return res;
  
}

template<typename T>
void
device_custom_reduce_atomics(T* arr, size_t size, T* out,
                             sycl::nd_item<3> item_ct1){
    T sum = 0.;
	//reduce multiple elements per thread
        const int tid =
          item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
        const int total_num_threads =
          item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);

        for (size_t i = tid; i < size; i += total_num_threads) {
		sum += arr[i];
	}

        sum = quad::warpReduceSum(sum, item_ct1);
        
    const int warpSize = 32;
        if ((item_ct1.get_local_id(2) & (warpSize - 1)) == 0) {
                
	  dpct::atomic_fetch_add(out, sum);
        }
}

template <typename T>
T
custom_reduce_atomics(T* arr, size_t size) {
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  T res = 0.;
	size_t num_threads = 256;
	size_t max_num_blocks = 1024;
        size_t num_blocks =
          std::min((size + num_threads - 1) / num_threads, max_num_blocks);
        T* out = quad::cuda_malloc<T>(1);
	quad::cuda_memcpy_to_device<T>(out, &res, 1);
	
	quad::cuda_memcpy_to_device<T>(out, &res, 1);
        
        q_ct1.parallel_for(
          sycl::nd_range(sycl::range(1, 1, num_blocks) *
                           sycl::range(1, 1, num_threads),
                         sycl::range(1, 1, num_threads)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                  device_custom_reduce_atomics(arr, size, out, item_ct1);
          }).wait();

        quad::cuda_memcpy_to_host<T>(&res, out, 1);
        sycl::free(out, q_ct1);
        return res;
}

template<typename T1, typename T2>

void
device_custom_inner_product_atomics(T1* arr1, T2* arr2, size_t size, T2* out,
                                    sycl::nd_item<3> item_ct1){
    T2 sum = 0.;
	//reduce multiple elements per thread
        const int tid =
          item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
        const int total_num_threads =
          item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);

        for (size_t i = tid; i < size; i += total_num_threads) {
		sum += arr1[i]*arr2[i];
	}

        sum = quad::warpReduceSum(sum, item_ct1);
        item_ct1.barrier();
        const int warpSize = 32;
        if ((item_ct1.get_local_id(2) & (warpSize - 1)) == 0) {
                
	  //dpct::atomic_fetch_add(out, sum);
        }
}

template <typename T1, typename T2>
T2
custom_inner_product_atomics(T1* arr1, T2* arr2, size_t size) {
	auto q_ct1 =  sycl::queue(sycl::gpu_selector());
	T2 res = 0.;
	size_t num_threads = 256;
	size_t max_num_blocks = 1024;
    size_t num_blocks =
    std::min((size + num_threads - 1) / num_threads, max_num_blocks);
    T2* out = quad::cuda_malloc<T2>(1);
	quad::cuda_memcpy_to_device<T2>(out, &res, 1);
        
	q_ct1.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, num_blocks) *
                        sycl::range(1, 1, num_threads),
                        sycl::range(1, 1, num_threads)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
			  
                device_custom_inner_product_atomics<T1, T2>(arr1, arr2, size, out, item_ct1);
          });
	}).wait_and_throw();
	
	
    quad::cuda_memcpy_to_host<T2>(&res, out, 1);
    sycl::free(out, q_ct1);
    return res;
}




template<typename T>
__inline__ T warpReduceMax(T val, sycl::nd_item<3> item_ct1)
{
    const unsigned int FULL_MASK = 0xffffffff;

    for (int mask = item_ct1.get_sub_group().get_local_range().get(0) / 2;
         mask > 0;
         mask /= 2)
    {
        
        val = sycl::max(
          sycl::permute_group_by_xor(item_ct1.get_sub_group(), val, mask), val);
    }
      
    return val;
}

template<typename T>
__inline__ T warpReduceMin(T val, sycl::nd_item<3> item_ct1)
{
    const unsigned int FULL_MASK = 0xffffffff;

    for (int mask = item_ct1.get_sub_group().get_local_range().get(0) / 2;
         mask > 0;
         mask /= 2)
    {
        
        val = sycl::min(
          sycl::permute_group_by_xor(item_ct1.get_sub_group(), val, mask), val);
    }
      
    return val;
}

template <typename T>
void
blockReduceMinMax(T& min, T& max, sycl::nd_item<3> item_ct1, T *shared_max,
                  T *shared_min)
{

    int lane = item_ct1.get_local_id(2) % 32; // 32 is for warp size
    int wid = item_ct1.get_local_id(2) >> 5 /* threadIdx.x / 32  */;

    min = warpReduceMin(min, item_ct1);
        max = warpReduceMax(max, item_ct1);

    if (lane == 0) {
      shared_min[wid] = min;
	  shared_max[wid] = max;
	  //printf("all warps blockReduceMinMax [%i](%i) min:%f\n", blockIdx.x, threadIdx.x, min);
    }
    
    item_ct1.barrier(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    min =
      (item_ct1.get_local_id(2) < (item_ct1.get_local_range().get(2) >> 5)) ?
        shared_min[lane] :
        std::numeric_limits<T>::max();
        max = (item_ct1.get_local_id(2) <
               (item_ct1.get_local_range().get(2) >> 5)) ?
                shared_max[lane] :
                0.;

    if (wid == 0){
                min = warpReduceMin(min, item_ct1);
                max = warpReduceMax(max, item_ct1);
        }
}

template<typename T>

void 
blocks_min_max(const T* __restrict__ input, const int size, T* min, T* max,
               sycl::nd_item<3> item_ct1,
               T* shared_max,
               T* shared_min)
{
        const int tid =
          item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
        const int total_num_threads =
          item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2);

    T localMax = 0;
    T localMin = std::numeric_limits<T>::max();
	
	for (int i = tid; i < size; i += total_num_threads){
        T val = input[tid];
		
        if (localMax < val){
            localMax = val;
        }
		
		if(localMin > val){
			localMin = val;
		}
    }

    blockReduceMinMax(localMin, localMax, item_ct1, shared_max, shared_min);

    if (item_ct1.get_local_id(2) == 0) {
        max[item_ct1.get_group(2)] = localMax;
                min[item_ct1.get_group(2)] = localMin;
    }
}

template<typename T>
void block0_min_max(T* mins, T* maxs, const int size, T* min, T* max,
                    sycl::nd_item<3> item_ct1,
                    T* shared_max,
                    T* shared_min)
{
        const int tid = item_ct1.get_local_id(2);

    T localMax = tid < size ? maxs[tid] : 0;
    T localMin = tid < size ? mins[tid] : std::numeric_limits<T>::max();

    blockReduceMinMax(localMin, localMax, item_ct1, shared_max, shared_min);

    if (item_ct1.get_local_id(2) == 0) {
        max[item_ct1.get_group(2)] = localMax;
                min[item_ct1.get_group(2)] = localMin;
                //printf("reducing the block results [%i] :%f,%f\n", blockIdx.x, localMax, localMin);
    }
}

template <typename T>
std::pair<T, T>
min_max(T* input, const int size) {
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  size_t num_threads = 256;
  auto device = q_ct1.get_device();
	size_t max_num_blocks = device.get_info<cl::sycl::info::device::max_work_group_size>();
        size_t num_blocks =
          std::min((size + num_threads - 1) / num_threads, max_num_blocks);

        T* block_mins = quad::cuda_malloc<T>(num_blocks);
	T* block_maxs = quad::cuda_malloc<T>(num_blocks);
	T* d_min = quad::cuda_malloc<T>(1);
	T* d_max = quad::cuda_malloc<T>(1);

        
        q_ct1.submit([&](sycl::handler& cgh) {
                sycl::accessor<T,
                               1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                  shared_max_acc_ct1(sycl::range(32), cgh);
                sycl::accessor<T,
                               1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                  shared_min_acc_ct1(sycl::range(32), cgh);

                cgh.parallel_for(
                  sycl::nd_range(sycl::range(1, 1, num_blocks) *
                                   sycl::range(1, 1, num_threads),
                                 sycl::range(1, 1, num_threads)),
                  [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                            blocks_min_max<T>(input,
                                              size,
                                              block_mins,
                                              block_maxs,
                                              item_ct1,
                                              shared_max_acc_ct1.get_pointer(),
                                              shared_min_acc_ct1.get_pointer());
                    });
	  }).wait();
        
        q_ct1.submit([&](sycl::handler& cgh) {
                sycl::accessor<T,
                               1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                  shared_max_acc_ct1(sycl::range(32), cgh);
                sycl::accessor<T,
                               1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                  shared_min_acc_ct1(sycl::range(32), cgh);

                cgh.parallel_for(
                  sycl::nd_range(
                    sycl::range(1, 1, std::max(num_blocks, (size_t)32)),
                    sycl::range(1, 1, std::max(num_blocks, (size_t)32))),
                  [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                            block0_min_max<T>(block_mins,
                                              block_maxs,
                                              num_blocks,
                                              d_min,
                                              d_max,
                                              item_ct1,
                                              shared_max_acc_ct1.get_pointer(),
                                              shared_min_acc_ct1.get_pointer());
                    });
	  }).wait();

        //dev_ct1.queues_wait_and_throw();

        T min = 0.;
	T max = 0.;
	
	quad::cuda_memcpy_to_host(&min, d_min, 1);
	quad::cuda_memcpy_to_host(&max, d_max, 1);

        sycl::free(block_mins, q_ct1);
        sycl::free(block_maxs, q_ct1);
        sycl::free(d_min, q_ct1);
        sycl::free(d_max, q_ct1);
        return {min, max};
}



template<typename T> 
void gpu_sum_scan_blelloch(T* const d_out,
    const T* const d_in,
    T* const d_block_sums,
    const size_t numElems,
    sycl::nd_item<3> item_ct1,
    T *dpct_local)
{
        auto s_out = (T*)dpct_local;

        // Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	//s_out[2 * threadIdx.x] = 0;
	//s_out[2 * threadIdx.x + 1] = 0;
        s_out[item_ct1.get_local_id(2)] = 0.;
        s_out[item_ct1.get_local_id(2) + item_ct1.get_local_range().get(2)] = 0.;

        
        item_ct1.barrier();

        //}
        unsigned int cpy_idx =
          2 * item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
        if (cpy_idx < numElems)
	{
                s_out[item_ct1.get_local_id(2)] = d_in[cpy_idx];
                if (cpy_idx + item_ct1.get_local_range().get(2) < numElems)
                        s_out[item_ct1.get_local_id(2) +
                              item_ct1.get_local_range().get(2)] =
                          d_in[cpy_idx + item_ct1.get_local_range().get(2)];
        }

        
        item_ct1.barrier();

        // Reduce/Upsweep step

	// 2^11 = 2048, the max amount of data a block can blelloch scan
	unsigned int max_steps = 11; 

    unsigned int r_idx = 0;
    unsigned int l_idx = 0;
    int sum = 0; // global sum can be passed to host if needed
    unsigned int t_active = 0;
    for (int s = 0; s < max_steps; ++s)
    {
        t_active = 0;

        // calculate necessary indexes
        // right index must be (t+1) * 2^(s+1)) - 1
        r_idx = ((item_ct1.get_local_id(2) + 1) * (1 << (s + 1))) - 1;
        if (/*r_idx >= 0 &&*/ r_idx < 2048)
            t_active = 1;

        if (t_active)
        {
            // left index must be r_idx - 2^s
            l_idx = r_idx - (1 << s);

            // do the actual add operation
            sum = s_out[l_idx] + s_out[r_idx];
        }
                
        item_ct1.barrier();

                if (t_active)
        s_out[r_idx] = sum;
                
          item_ct1.barrier();
        }

    // Copy last element (total sum of block) to block sums array
    // Then, reset last element to operation's identity (sum, 0)
        if (item_ct1.get_local_id(2) == 0)
        {
                d_block_sums[item_ct1.get_group(2)] = s_out[r_idx];
                s_out[r_idx] = 0.;
        }

        
        item_ct1.barrier();

        // Downsweep step

    for (int s = max_steps - 1; s >= 0; --s)
    {
        // calculate necessary indexes
        // right index must be (t+1) * 2^(s+1)) - 1
        r_idx = ((item_ct1.get_local_id(2) + 1) * (1 << (s + 1))) - 1;
        if (/*r_idx >= 0 &&*/ r_idx < 2048)
        {
            t_active = 1;
        }

        T r_cpy = 0.;
        T lr_sum = 0.;
        if (t_active)
        {
            // left index must be r_idx - 2^s
            l_idx = r_idx - (1 << s);

            // do the downsweep operation
            r_cpy = s_out[r_idx];
            lr_sum = s_out[l_idx] + s_out[r_idx];
        }
          
                item_ct1.barrier();

        if (t_active)
        {
            s_out[l_idx] = r_cpy;
            s_out[r_idx] = lr_sum;
        }
            
                item_ct1.barrier();
        }

    // Copy the results to global memory
    //if (2 * glbl_tid < numElems)
    //{
    //	d_out[2 * glbl_tid] = s_out[2 * threadIdx.x];
    //	if (2 * glbl_tid + 1 < numElems)
    //		d_out[2 * glbl_tid + 1] = s_out[2 * threadIdx.x + 1];
    //}
    if (cpy_idx < numElems)
    {
                d_out[cpy_idx] = s_out[item_ct1.get_local_id(2)];
                if (cpy_idx + item_ct1.get_local_range().get(2) < numElems)
                        d_out[cpy_idx + item_ct1.get_local_range().get(2)] =
                          s_out[item_ct1.get_local_id(2) +
                                item_ct1.get_local_range().get(2)];
    }
}

template<typename T>
void gpu_add_block_sums(T* const d_out,
    const T* const d_in,
    T* const d_block_sums,
    const size_t numElems,
    sycl::nd_item<3> item_ct1)
{
        T d_block_sum_val = d_block_sums[item_ct1.get_group(2)];

    // Simple implementation's performance is not significantly (if at all)
    //  better than previous verbose implementation
        unsigned int cpy_idx =
          2 * item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);
        if (cpy_idx < numElems)
        {
            d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
                if (cpy_idx + item_ct1.get_local_range().get(2) < numElems)
                        d_out[cpy_idx + item_ct1.get_local_range().get(2)] =
                          d_in[cpy_idx + item_ct1.get_local_range().get(2)] +
                          d_block_sum_val;
        }
}

// Modified version of Mark Harris' implementation of the Blelloch scan
//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
//this macro is only used for prefix scan

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

#define MAX_BLOCK_SZ 256
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

template<typename T>
void gpu_prescan(T* const d_out,
    const T* const d_in,
    T* const d_block_sums,
    const unsigned int len,
    const unsigned int shmem_sz,
    const unsigned int max_elems_per_block,
    sycl::nd_item<3> item_ct1,
    T *dpct_local)
{
    // Allocated on invocation
        auto s_out = (int*)dpct_local;

        int thid = item_ct1.get_local_id(2);
        int ai = thid;
        int bi = thid + item_ct1.get_local_range().get(2);

        // Zero out the shared memory
    // Helpful especially when input size is not power of two
    s_out[thid] = 0.;
        s_out[thid + item_ct1.get_local_range().get(2)] = 0;
        // If CONFLICT_FREE_OFFSET is used, shared memory
    //  must be a few more than 2 * blockDim.x
    if (thid + max_elems_per_block < shmem_sz)
        s_out[thid + max_elems_per_block] = 0.;

        item_ct1.barrier();

        // Copy d_in to shared memory
    // Note that d_in's elements are scattered into shared memory
    //  in light of avoiding bank conflicts
        unsigned int cpy_idx = max_elems_per_block * item_ct1.get_group(2) +
                               item_ct1.get_local_id(2);
        if (cpy_idx < len)
        {
            s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
                if (cpy_idx + item_ct1.get_local_range().get(2) < len)
                        s_out[bi + CONFLICT_FREE_OFFSET(bi)] =
                          d_in[cpy_idx + item_ct1.get_local_range().get(2)];
        }

	// For both upsweep and downsweep:
	// Sequential indices with conflict free padding
	//  Amount of padding = target index / num banks
	//  This "shifts" the target indices by one every multiple
	//   of the num banks
	// offset controls the stride and starting index of 
	//  target elems at every iteration
	// d just controls which threads are active
	// Sweeps are pivoted on the last element of shared memory

	// Upsweep/Reduce step
    int offset = 1;
    for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
    {
            
                item_ct1.barrier();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_out[bi] += s_out[ai];
        }
        offset <<= 1;
    }

    // Save the total sum on the global block sums array
    // Then clear the last element on the shared memory
    if (thid == 0) 
    {
        d_block_sums[item_ct1.get_group(2)] =
                  s_out[max_elems_per_block - 1 +
                        CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
                s_out[max_elems_per_block - 1 
            + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
    }

    // Downsweep step
    for (int d = 1; d < max_elems_per_block; d <<= 1)
    {
        offset >>= 1;
            
                item_ct1.barrier();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int temp = s_out[ai];
            s_out[ai] = s_out[bi];
            s_out[bi] += temp;
        }
    }
        
        item_ct1.barrier();

        // Copy contents of shared memory to global memory
	if (cpy_idx < len)
	{
		d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
                if (cpy_idx + item_ct1.get_local_range().get(2) < len)
                        d_out[cpy_idx + item_ct1.get_local_range().get(2)] =
                          s_out[bi + CONFLICT_FREE_OFFSET(bi)];
        }
}

template<typename T>
void sum_scan_blelloch(T* const d_out,
	const T* const d_in,
	const size_t numElems)
{
  auto q_ct1 =  sycl::queue(sycl::gpu_selector());
  // Zero out d_out
        
        q_ct1.memset(d_out, 0., numElems * sizeof(T)).wait();

        // Set up number of threads and blocks
    
    unsigned int block_sz = MAX_BLOCK_SZ / 2;
    unsigned int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

    // If input size is not power of two, the remainder will still need a whole block
    // Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
    //unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
    // UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically  
    //  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
    unsigned int grid_sz = numElems / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (numElems % max_elems_per_block != 0) 
        grid_sz += 1;

    // Conflict free padding requires that shared memory be more than 2 * block_sz
    unsigned int shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    T* d_block_sums;
    d_block_sums = sycl::malloc_device<T>(grid_sz, q_ct1);
    q_ct1.memset(d_block_sums, 0., sizeof(T) * grid_sz).wait();

    
        q_ct1.submit([&](sycl::handler& cgh) {
                sycl::accessor<T,
                               1,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                  dpct_local_acc_ct1(sycl::range(sizeof(T) * shmem_sz), cgh);

                cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, grid_sz) *
                                                  sycl::range(1, 1, block_sz),
                                                sycl::range(1, 1, block_sz)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                         gpu_prescan<T>(
                                           d_out,
                                           d_in,
                                           d_block_sums,
                                           numElems,
                                           shmem_sz,
                                           max_elems_per_block,
                                           item_ct1,
                                           dpct_local_acc_ct1.get_pointer());
                                 });
        }).wait();

        // Sum scan total sums produced by each block
    // Use basic implementation if number of total sums is <= 2 * block_sz
    //  (This requires only one block to do the scan)
    if (grid_sz <= max_elems_per_block)
    {
        T* d_dummy_blocks_sums;

                d_dummy_blocks_sums = sycl::malloc_device<T>(1, q_ct1);
                
                q_ct1.memset(d_dummy_blocks_sums, 0, sizeof(T)).wait();
                //gpu_sum_scan_blelloch<<<1, block_sz, sizeof(unsigned int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
                
                q_ct1.submit([&](sycl::handler& cgh) {
                        sycl::accessor<T,
                                       1,
                                       sycl::access_mode::read_write,
                                       sycl::access::target::local>
                          dpct_local_acc_ct1(
                            sycl::range(sizeof(T) * shmem_sz), cgh);

                        cgh.parallel_for(
                          sycl::nd_range(sycl::range(1, 1, block_sz),
                                         sycl::range(1, 1, block_sz)),
                          [=](sycl::nd_item<3> item_ct1) {
                                  gpu_prescan<T>(d_block_sums,
                                              d_block_sums,
                                              d_dummy_blocks_sums,
                                              grid_sz,
                                              shmem_sz,
                                              max_elems_per_block,
                                              item_ct1,
                                              dpct_local_acc_ct1.get_pointer());
                          });
                }).wait();
                
                sycl::free(d_dummy_blocks_sums, q_ct1);
        }
	// Else, recurse on this same function as you'll need the full-blown scan
	//  for the block sums
	else
	{
        T* d_in_block_sums;

        d_in_block_sums = sycl::malloc_device<T>(grid_sz, q_ct1);
               
        q_ct1.memcpy(d_in_block_sums, d_block_sums, sizeof(T) * grid_sz).wait();
        sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
        sycl::free(d_in_block_sums, q_ct1);
        }
	
	//// Uncomment to examine block sums
	//unsigned int* h_block_sums = new unsigned int[grid_sz];
	//(cudaMemcpy(h_block_sums, d_block_sums, sizeof(unsigned int) * grid_sz, cudaMemcpyDeviceToHost));
	//std::cout << "Block sums: ";
	//for (int i = 0; i < grid_sz; ++i)
	//{
	//	std::cout << h_block_sums[i] << ", ";
	//}
	//std::cout << std::endl;
	//std::cout << "Block sums length: " << grid_sz << std::endl;
	//delete[] h_block_sums;

	// Add each block's total sum to its scan output
	// in order to get the final, global scanned array
        
        q_ct1.parallel_for(
          sycl::nd_range(sycl::range(1, 1, grid_sz) *
                           sycl::range(1, 1, block_sz),
                         sycl::range(1, 1, block_sz)),
          [=](sycl::nd_item<3> item_ct1) {
                  gpu_add_block_sums(
                    d_out, d_out, d_block_sums, numElems, item_ct1);
          }).wait();

        
        ((sycl::free(d_block_sums, q_ct1), 0));
}


#endif
