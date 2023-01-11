#include <iostream>
#include <array>
#include <cuda.h>
#include <numeric>
#include <vector>

template<typename T>
T*
copy_to_host(T* dest, T* src, size_t size){
    auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyDeviceToHost);
    if(rc != cudaSuccess)
        throw std::bad_alloc();
	return dest;
}

template<class T>
T*
cuda_malloc(size_t size){
    T* temp;  
    auto rc = cudaMalloc((void**)&temp, sizeof(T) * size);
    if (rc != cudaSuccess){
      printf("device side\n");
      throw std::bad_alloc();
    }
    return temp;
}
  
template<typename T>
cudaError_t
copy_to_device(T* dest, T* src, size_t size){
    cudaError_t rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
    if(rc != cudaSuccess)
        throw std::bad_alloc();
	return rc;
}

template<typename T>
T*
alloc_and_copy_to_device(T* src, size_t size){
	T* tmp = cuda_malloc<T>(size);
	cudaError_t rc = copy_to_device<T>(tmp, src, size);
	if(rc != cudaSuccess)
		throw std::runtime_error("Bad copy to device");
	return tmp;
}

__global__ 
void
atomic_addition(double* src, double* out, size_t size){
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t total_num_threads = gridDim.x * blockDim.x; 
	
	for(size_t i = tid; i < size; i += total_num_threads){
		for(int i=0; i < 8;++i)
			atomicAdd(&out[threadIdx.x], src[i]);
		//out[threadIdx.x] += src[i];
	}
}

int main(){
    
	const size_t num_threads = 64;

	std::vector<double> src;
	src.resize(32768 * 1025 * 2);
	std::iota(src.begin(), src.end(), 1.);
	
	std::array<double, num_threads> output = {0.};
	
	std::cout<<"size:"<<src.size()<<std::endl;
	std::cout<<"Memory:"<<src.size()*8/1e9<<"GB\n";
	
	double* d_src = alloc_and_copy_to_device<double>(src.data(), src.size());
	double* d_output = alloc_and_copy_to_device<double>(output.data(), output.size());
	
	size_t num_blocks = src.size()/num_threads;
		
	atomic_addition<<<num_blocks, num_threads>>>(d_src, d_output, src.size());
	cudaDeviceSynchronize();
	
	copy_to_host(output.data(), d_output, output.size());
	
	for(int i = 0; i < output.size(); ++i)
		printf("output %i, %e\n", i, output[i]);
	
	cudaFree(d_src);
	cudaFree(d_output);
    return 0;
}

