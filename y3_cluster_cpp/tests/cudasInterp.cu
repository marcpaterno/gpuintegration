#include "catch2/catch.hpp"

#include "modules/sigma_miscent_y1_scalarintegrand.hh"
#include "../cudaCuhre/quad/util/cudaArray.cuh"

#include <iostream>
#include <chrono>					
#include "utils/str_to_doubles.hh"  
#include <vector> 					

#include <fstream>
#include <stdexcept>
#include <string>
#include <array>

#include "models/int_lc_lt_des_t.hh"
#include "models/omega_z_des.hh"
#include "models/int_zo_zt_des_t.hh"
#include "models/mor_des_t.hh"
#include "models/roffset_t.hh"
#include "models/dv_do_dz_t.hh"
#include "models/lo_lc_t.hh"
//using namespace y3_cluster;

//GPU integrator headers
#include "quad/GPUquad/Cuhre.cuh"
#include "quad/quad.h"
#include "quad/util/Volume.cuh"
#include "quad/util/cudaUtil.h"
//#include "function.cuh"
//CPU integrator headers
#include "cuba.h"
#include "cubacpp/cuhre.hh"
#include "vegas.h"
//#include "RZU.cuh"
#include <limits>
namespace quad {
	
	__device__ __host__
	inline double
	gaussian(double x, double mu, double sigma)
	{
		double const z = (x - mu) / sigma;
		return exp(-z * z / 2.) * 0.3989422804014327 / sigma;
	}
	
	class Managed 
	{
	 public:
	  void *operator new(size_t len) {
		void *ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	  }

	  void operator delete(void *ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	  }
	};	
	
	class Interp2D : public Managed{
	  public:
		__host__ __device__
		Interp2D(){}
		//change names to xs, ys, zs to fit with y3_cluster_cpp::Interp2D
		double* interpT;
		double* interpR;
		double* interpC;
		size_t _rows;
		size_t _cols;
		
		~Interp2D(){
			//cudaFree(interpT);
			//cudaFree(interpR);
			//cudaFree(interpC);
		}
		
		void Alloc(size_t cols, size_t rows){
			_rows = rows;
			_cols = cols;
			cudaMallocManaged((void**)&interpR, sizeof(double)*_rows);
			cudaMallocManaged((void**)&interpC, sizeof(double)*_cols);
			cudaMallocManaged((void**)&interpT, sizeof(double)*_rows*_cols);
		}
		
		template<size_t M, size_t N>
		Interp2D(std::array<double, M> const& xs, std::array<double, N> const& ys, std::array<double, (N)*(M)> const& zs)
		{
			Alloc(M, N);
			memcpy(interpR, ys.data(), sizeof(double)*N);
			memcpy(interpC, xs.data(), sizeof(double)*M);
			memcpy(interpT, zs.data(), sizeof(double)*N*M);
		}
		
		Interp2D(double* xs, double* ys, double* zs, size_t cols, size_t rows)
		{
			Alloc(cols, rows);
			memcpy(interpR, ys, sizeof(double)*rows);
			memcpy(interpC, xs, sizeof(double)*cols);
			memcpy(interpT, zs, sizeof(double)*rows*cols);
		}
		
		//Interp2D(std::vector<double>&& xs, std::vector<double>&& ys, std::vector<std::vector<double>> const& zs)
		//Look at mor_des_t model, is the data in .cc file given in wrong format?
		template<size_t M, size_t N>
		Interp2D(std::array<double, M> xs, std::array<double, N> ys, std::array<std::array<double, N>, M> zs)
		{		
			Alloc(M, N);
			memcpy(interpR, ys.data(), sizeof(double)*N);
			memcpy(interpC, xs.data(), sizeof(double)*M);
			
			
			for (std::size_t i = 0; i < M; ++i) {
				std::array<double, N> const& row = zs[i];
				for (std::size_t j = 0; j < N; ++j) {
					interpT[i + j * M] = row[j];
					//interpT[i + j * N] = zs[i][j];
					//printf("custom: row[%lu]:%f -> InterpT[%lu]\n", j, row[j], i+j*N);
				}
			}
			//printf("%f, %f, %f, %f\n", min_x(), max_x(), min_y(), max_y());
	    }
		
		__device__ 
		bool AreNeighbors(const double val, double* arr, const size_t leftIndex, const size_t RightIndex) const{
			//printf("[%i](%i) evaluating neighbors %f, %f against val:%f index:%lu\n", blockIdx.x, threadIdx.x, arr[leftIndex], arr[RightIndex], val, leftIndex);
			if(arr[leftIndex] <= val && arr[RightIndex] >= val)
				return true;
			return false;
		}
		
		friend std::istream&
		operator>>(std::istream& is, Interp2D& interp)
		{
		  assert(is.good());
		  std::string buffer;
		  std::getline(is, buffer);
		  std::vector<double> xs = cosmosis::str_to_doubles(buffer);
		  std::getline(is, buffer);
		  std::vector<double> ys = cosmosis::str_to_doubles(buffer);
		  std::getline(is, buffer);
		  std::vector<double> zs  = cosmosis::str_to_doubles(buffer);
		  
		  cudaMallocManaged((void**)&(*&interp), sizeof(Interp2D));
		  
		  interp._cols = xs.size();
		  interp._rows = ys.size();
		  
		  
		  cudaMallocManaged((void**)&interp.interpR, sizeof(double)*ys.size());
		  cudaDeviceSynchronize();
		  cudaMallocManaged((void**)&interp.interpC, sizeof(double)*xs.size());
		  cudaDeviceSynchronize();
		  cudaMallocManaged((void**)&interp.interpT, sizeof(double)*zs.size());
		  cudaDeviceSynchronize();
		  
		  memcpy(interp.interpR, ys.data(), sizeof(double)*ys.size());
		  memcpy(interp.interpC, xs.data(), sizeof(double)*xs.size());
		  memcpy(interp.interpT, zs.data(), sizeof(double)*zs.size());
		  
		  return is;
		}
		
		Interp2D(const Interp2D &source) {
			//cudaMallocManaged((void**)source, sizeof(Interp2D));
			Alloc(source._cols, source._rows);
			interpT = source.interpT;
			interpC = source.interpC;
			interpR = source.interpR;
			_cols = source._cols;
			_rows = source._rows;
			//printf("Interp2D copy constructor called: %lu vs %lu\n", _cols, source._cols);
		} 
		
		__device__
		void FindNeighbourIndices(const double val, double* arr, const size_t size, size_t& leftI, size_t& rightI) const{
	
			size_t currentIndex = size/2;
			leftI = 0;
			rightI = size - 1;

			//while(currentIndex != 0 && currentIndex != lastIndex){
			while(leftI<=rightI){
				//currentIndex = leftI + (rightI - leftI)/2;
				currentIndex = (rightI+leftI)*0.5;
				/*if(val == 244.)
					printf("[%i](%i) looking for %f l:%lu r:%lu checking:%f & %f currentIndex:%lu\n", blockIdx.x, 
																								  threadIdx.x,  
																								  val,
																								  leftI, 
																								  rightI,  
																								  arr[leftI], 
																								  arr[currentIndex],
																								  currentIndex);*/
				/*printf("[%i](%i) looking for %f l:%lu r:%lu checking:%f & %f currentIndex:%lu\n", blockIdx.x, 
																								  threadIdx.x,  
																								  val,
																								  leftI, 
																								  rightI,  
																								  arr[leftI], 
																								  arr[rightI],
																								  currentIndex);*/
				//if(AreNeighbors(val, arr, currentIndex-1, currentIndex)){
				if(AreNeighbors(val, arr, currentIndex, currentIndex+1)){
					//leftI = currentIndex -1;
					//rightI = currentIndex;
					leftI = currentIndex;
					rightI = currentIndex+1;
					//printf("returning indices :%lu, %lu\n", leftI, rightI);
					return;
				}
				
				if(arr[currentIndex] > val){
					rightI = currentIndex;
				}
				else{
					leftI = currentIndex;
				}
				//printf("[%i](%i) currentIndex:%lu\n", currentIndex);
			}
		}
		
		__device__ double
		operator()(double x, double y) const
		{
		 //printf("custom: interpolating on %.20f, %.20f\n", x, y);
		 // for(int i=0; i< _rows*_cols; ++i)
		//	printf("interpT[%i]:%f\n", i, interpT[i]);
	
		  //y1, y2, x1, x2, are the indices of where to find the four neighbouring points in the z-table
		  size_t y1 = 0, y2 = 0;
		  size_t x1 = 0, x2 = 0;
			
		  FindNeighbourIndices(y, interpR, _rows, y1, y2);
		  FindNeighbourIndices(x, interpC, _cols, x1, x2);
		  //printf("interpolation indices(%lu, %lu), (%lu, %lu)\n", x1, x2, y1, y2);
		  //this is how  zij is accessed by gsl2.6 Interp2D i.e. zij = z[j*xsize+i], where i=0,...,xsize-1, j=0, ..., ysize-1
		  const double q11 = interpT[y1*_cols + x1];
		  const double q12 = interpT[y2*_cols + x1];
		  const double q21 = interpT[y1*_cols + x2];
		  const double q22 = interpT[y2*_cols + x2];
		  
		  const double x1_val = interpC[x1];
		  const double x2_val = interpC[x2];
		  const double y1_val = interpR[y1];
		  const double y2_val = interpR[y2];
		  
		  const double f_x_y1 = q11*(x2_val-x)/(x2_val-x1_val) + q21*(x-x1_val)/(x2_val-x1_val);
		  const double f_x_y2 = q12*(x2_val-x)/(x2_val-x1_val) + q22*(x-x1_val)/(x2_val-x1_val);
		  
		  //printf("indices of neighbors:x1:%lu, x2:%lu, y1:%lu, y2:%lu\n", x1, x2, y1, y2);
		  //printf("q11:%f, q12:%f, q21:%f, q22:%f\n", q11, q12, q21, q22);
		  
		  double f_x_y = 0.;
		  f_x_y = f_x_y1*(y2_val-y)/(y2_val-y1_val) + f_x_y2*(y-y1_val)/(y2_val-y1_val); 
		  return f_x_y;
		}
		
		__device__ double
		min_x() const{ 
		//printf("min_x:%f\n", interpC[0]);
		//printf("min_x:\n");
		//printf("min _cols:%lu\n", _cols);
		//printf("interpC[0]:%lu\n", interpC[0]);
		return interpC[0]; }
		
		__device__ double
		max_x() const{ 
		//printf("cols:%lu max_x:%f\n", _cols, interpC[_cols-1]);
		//printf("max_x\n");
		return interpC[_cols-1]; }
		
		__device__   double
		min_y() const{ 
		//printf("min_y\n");
		//printf("min_y\n");
		return interpR[0]; }
		
		__device__ double
		max_y() const{ 
		//printf("max_y\n");
		return interpR[_rows-1]; }
		
		__device__  double
		do_clamp(double v, double lo, double hi) const
		{
			//printf("inside do_clamp\n");
			//printf("[%i](%i)do clamp (%f, %f, %f)\n", blockIdx.x, threadIdx.x, v, lo, hi);
			assert(!(hi < lo));
			return (v < lo) ? lo : (hi < v) ? hi : v;
		}
		
		__device__ double
		eval(double x, double y) const
		{
		  //printf("[%i](%i)evaluate interpolator at  (%f, %f)\n", blockIdx.x, threadIdx.x, x, y);
		  return this->operator()(x, y);
		};
		
		__device__ 
		double
		clamp(double x, double y) const
		{
	      //printf("[%i](%i)clamp (%f, %f)\n", blockIdx.x, threadIdx.x, x, y);
		  return eval(do_clamp(x, min_x(), max_x()), do_clamp(y, min_y(), max_y()));
		}
	  };
	
	class Interp1D : public Managed{
	  public:
		__host__ __device__
		Interp1D(){}
		//change names to xs, ys, zs to fit with y3_cluster_cpp::Interp2D
		double* interpT;
		double* interpC;
		size_t _cols;
		
		~Interp1D(){
			//cudaFree(interpT);
			//cudaFree(interpC);
		}
		
		void Alloc(size_t cols){
			_cols = cols;
			cudaMallocManaged((void**)&interpC, sizeof(double)*_cols);
			cudaMallocManaged((void**)&interpT, sizeof(double)*_cols);
		}
		
		template<size_t M>
		Interp1D(std::array<double, M> const& xs, std::array<double,M> const& zs)
		{
			Alloc(M);
			memcpy(interpC, xs.data(), sizeof(double)*M);
			memcpy(interpT, zs.data(), sizeof(double)*M);
		}
		
		Interp1D(double* xs, double* ys, double* zs, size_t cols)
		{
			Alloc(cols);
			memcpy(interpC, xs, sizeof(double)*cols);
			memcpy(interpT, zs, sizeof(double)*cols);
		}
		
		__device__ 
		bool AreNeighbors(const double val, double* arr, const size_t leftIndex, const size_t RightIndex) const{
			if(arr[leftIndex] <= val && arr[RightIndex] >= val)
				return true;
			return false;
		}
		
		friend std::istream&
		operator>>(std::istream& is, Interp1D& interp)
		{
		  assert(is.good());
		  std::string buffer;
		  std::getline(is, buffer);
		  std::vector<double> xs = cosmosis::str_to_doubles(buffer);
		  std::getline(is, buffer);
		  std::vector<double> zs  = cosmosis::str_to_doubles(buffer);
		  
		  cudaMallocManaged((void**)&(*&interp), sizeof(Interp1D));
		  cudaDeviceSynchronize();
		  
		  interp._cols = xs.size();
		  
		  cudaMallocManaged((void**)&interp.interpC, sizeof(double)*xs.size());
		  cudaDeviceSynchronize();
		  cudaMallocManaged((void**)&interp.interpT, sizeof(double)*zs.size());
		  cudaDeviceSynchronize();
		  
		  memcpy(interp.interpC, xs.data(), sizeof(double)*xs.size());
		  memcpy(interp.interpT, zs.data(), sizeof(double)*zs.size());
		  
		  return is;
		}
		
		Interp1D(const Interp1D &source) {
			Alloc(source._cols);
			interpT = source.interpT;
			interpC = source.interpC;
			_cols = source._cols;
		} 
		
		__device__
		void FindNeighbourIndices(const double val, double* arr, const size_t size, size_t& leftI, size_t& rightI) const{

			size_t currentIndex = size/2;
			leftI = 0;
			rightI = size - 1;

			//while(currentIndex != 0 && currentIndex != lastIndex){
			while(leftI<=rightI){
				//currentIndex = leftI + (rightI - leftI)/2;
				currentIndex = (rightI + leftI)*0.5;
				if(AreNeighbors(val, arr, currentIndex, currentIndex+1)){
					leftI = currentIndex;
					rightI = currentIndex+1;
					return;
				}
				
				if(arr[currentIndex] > val){
					rightI = currentIndex;
				}
				else{
					leftI = currentIndex;
				}
			}
		}
		
		__device__ double
		operator()(double x) const
		{
		  size_t x0_index = 0, x1_index = 0;
		  FindNeighbourIndices(x, interpC, _cols, x0_index, x1_index);
		  const double y0 = interpT[x0_index];
		  const double y1 = interpT[x1_index];
		  const double x0 = interpC[x0_index];
		  const double x1 = interpC[x1_index];
		  const double y = (y0*(x1 - x) + y1*(x - x0))/(x1 - x0);
		  return y;
		}
		
		__device__ double
		min_x() const{ return interpC[0]; }
		
		__device__ double
		max_x() const{ return interpC[_cols-1]; }
		
		__device__  double
		do_clamp(double v, double lo, double hi) const
		{
			assert(!(hi < lo));
			return (v < lo) ? lo : (hi < v) ? hi : v;
		}
		
		__device__ double
		eval(double x) const
		{
		  return this->operator()(x);
		};
		
		__device__ 
		double
		clamp(double x) const
		{
		  return eval(do_clamp(x, min_x(), max_x()));
		}
	  };
	
	template<size_t Order>
	class polynomial{
	  private:
		const gpu::cudaArray<double, Order> coeffs;
	  public:
	  __host__ __device__
		polynomial(gpu::cudaArray<double, Order> coeffs) : coeffs(coeffs) {}
		
		__host__ __device__
		constexpr double
		operator()(const double x) const{
			double out = 0.0;
			for (auto i = 0u; i < Order; i++)
				out = coeffs[i] + x * out;
			return out;
		}
	};
}

template <class T>
class hmf_t {
	  public:
		hmf_t() = default;
		
		__host__
		hmf_t(typename T::Interp2D* nmz, double s, double q)
		  : _nmz(nmz), _s(s), _q(q)
		{}
		
		using doubles = std::vector<double>;
		
		//ADD DATABLOCK CONSTRUCTOR
		__device__ __host__
		double
		operator()(double lnM, double zt) const{
		  return _nmz->clamp(lnM, zt) *
				 (_s * (lnM * 0.4342944819 - 13.8124426028) + _q);
		}
		
		friend std::ostream&
		operator<<(std::ostream& os, hmf_t const& m){
		  auto const old_flags = os.flags();
		  os << std::hexfloat;
		  os << *(m._nmz) << '\n' << m._s << ' ' << m._q;
		  os.flags(old_flags);
		  return os;
		}
		
		friend std::istream&
		operator>>(std::istream& is, hmf_t& m){
		  assert(is.good());
		  //doing the line below instead //auto table = std::make_shared<typename T::Interp2D>();
		  typename T::Interp2D *table = new typename T::Interp2D;
		  is >> *table;
		  
		  std::string buffer;
		  std::getline(is, buffer);
		  std::vector<double> const vals_read = cosmosis::str_to_doubles(buffer);
		  if (vals_read.size() == 2)
		  {
			m = hmf_t(table, vals_read[0], vals_read[1]);
		  }
		  else
		  {
			is.setstate(std::ios_base::failbit);
		  };
		  return is;
		}
		
	  private:
		typename T::Interp2D* _nmz;
		//std::shared_ptr<typename T::Interp2D const> _nmz;
		
		double _s = 0.0;
		double _q = 0.0;
};

struct GPU {
  template<size_t order>
  using polynomial = quad::polynomial<order>;
  typedef quad::Interp2D Interp2D;
  typedef quad::Interp1D Interp1D;
  //typedef quad::ez ez;
};

struct CPU {
  template<size_t order>
  using polynomial = y3_cluster::polynomial<order>;
  typedef y3_cluster::Interp2D Interp2D;
  typedef y3_cluster::Interp1D Interp1D;
};

template<typename T>
__global__ 
void
testKernel(T* model, double x, double y, double* result){
	*result = model->operator()(x, y);
}

template<typename T>
__global__ 
void
testKernel(T* model, double x, double y, double z, double* result){
	*result = model->operator()(x, y, z);
}

template<typename T>
__global__ 
void
testKernel(T* model, double x, double* result){
	*result = model->operator()(x);
}

template<typename T>
__global__ 
void
testKernel(T* model, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double* result){
	//printf("From kernel:%a\n", model->operator()(x1, x2, x3, x4, x5, x6, x7));
	*result = model->operator()(x1, x2, x3, x4, x5, x6, x7);
	//printf("Interpolator cols:%lu\n", model->mor.sig_interp->_cols);
	//printf("(%i):%f\n", threadIdx.x, *result);
}

template <class M>
M
make_from_file(char const* filename)
{
  static_assert(std::is_default_constructible<M>::value, "Type must be default constructable");
  char const* basedir = std::getenv("Y3_CLUSTER_CPP_DIR");
  if (basedir == nullptr) throw std::runtime_error("Y3_CLUSTER_CPP_DIR was not defined\n");
  std::string fname(basedir);
  fname += '/';
  fname += filename;
  std::ifstream in(fname);
  if (!in) {
    std::string msg("Failed to open file: ");
    msg += fname;
    throw std::runtime_error(msg);
  }
  M result;
  in >> result;
  return result;
}

template<class T>
class MockIntegrand{
	public:
		hmf_t<T> modelA;
		hmf_t<T> modelB;
	    
		__device__ __host__
		double 
		operator()(double x, double y){
			return modelA(x,y) + modelB(x,y);
		}
};

/*TEST_CASE("HMF_t CONDITIONAL MODEL EXECUTION")
{
	double const zt = 0x1.cccccccccccccp-2;
	double const lnM = 0x1.0cp+5;
	
	hmf_t<CPU> hmf  = make_from_file<hmf_t<CPU>>("data/HMF_t.dump");
	hmf_t<GPU> hmf2 = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
	
	SECTION("SAME BEHAVIOR WITH <GPU> OBJECT")
	{
		CHECK(hmf2(lnM,  zt) == hmf(lnM,  zt));
	}
	
	SECTION("SAME BEHAVIOR WITH <GPU> OBJECT ON GPU")
	{
		hmf_t<GPU> *dhmf2;
		cudaMallocManaged((void**)&dhmf2, sizeof(hmf_t<GPU>));
		cudaDeviceSynchronize();
		memcpy(dhmf2, &hmf2, sizeof(hmf_t<GPU>));
		
		double* result;
		cudaMallocManaged((void**)&result, sizeof(double));
		
		testKernel<hmf_t<GPU>><<<1,1>>>(dhmf2, lnM, zt, result);
		cudaDeviceSynchronize();
		CHECK(dhmf2->operator()(lnM,  zt) == hmf(lnM,  zt));
		CHECK(*result == hmf(lnM,  zt));
		
		cudaFree(dhmf2);
		cudaFree(result);
	}
	
	SECTION("MOCK INTEGRAL WITH TWO IDENTICAL MODELS, EACH WITH EACH OWN INTERP2D")
	{
		MockIntegrand<GPU> integrand;
		integrand.modelA = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
		integrand.modelB = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
		
		MockIntegrand<GPU> *d_integrand;
		cudaMallocManaged((void**)&d_integrand, sizeof(MockIntegrand<GPU>));
		cudaDeviceSynchronize();
		memcpy(d_integrand, &integrand, sizeof(MockIntegrand<GPU>));
		
		double* result;
		cudaMallocManaged((void**)&result, sizeof(double));
		
		testKernel<MockIntegrand<GPU>><<<1,1>>>(d_integrand, lnM, zt, result);
		cudaDeviceSynchronize();
		CHECK((double)(*result)/2 == hmf(lnM,  zt));
		
		cudaFree(d_integrand);
		cudaFree(result);
	}
}*/

  std::array<double, 16> constexpr zt_bins = {
											  0.000000,
                                              0.050000,
                                              0.100000,
                                              0.150000,
                                              0.200000,
                                              0.250000,
                                              0.300000,
                                              0.350000,
                                              0.400000,
                                              0.450000,
                                              0.500000,
                                              0.550000,
                                              0.600000,
                                              0.650000,
                                              0.700000,
                                              0.750000};

  std::array<double, 60> constexpr lt_bins = {
    1.000000,   2.000000,   3.000000,   4.000000,   5.000000,   6.000000,
    7.000000,   8.000000,   9.000000,   10.000000,  11.000000,  12.000000,
    13.000000,  14.000000,  15.000000,  16.000000,  17.000000,  18.000000,
    19.000000,  20.000000,  21.000000,  23.000000,  25.000000,  27.000000,
    29.000000,  31.000000,  33.000000,  35.000000,  37.000000,  39.000000,
    41.000000,  43.000000,  45.000000,  47.000000,  49.000000,  51.000000,
    53.000000,  55.000000,  57.000000,  59.000000,  63.000000,  67.000000,
    71.000000,  75.000000,  79.000000,  83.000000,  87.000000,  91.000000,
    95.000000,  99.000000,  105.000000, 115.000000, 125.000000, 135.000000,
    145.000000, 155.000000, 165.000000, 175.000000, 185.000000, 195.000000};

  std::array<double, 960> constexpr lambda0_arr = {
    0.000000, 0.000000, 0.000000, 0.000090, 0.000000, 0.000000, 0.000000,
    0.001465, 0.003800, 0.006993, 0.013266, 0.025400, 0.046494, 0.080917,
    0.133356, 0.208246, 0.302287, 0.403247, 0.498395, 0.575000, 0.622843,
    0.634095, 0.549647, 0.416874, 0.285337, 0.175278, 0.095138, 0.043228,
    0.017411, 0.012024, 0.013102, 0.013077, 0.012081, 0.010578, 0.008983,
    0.007425, 0.005931, 0.004532, 0.003257, 0.002131, 0.000318, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000090, 0.000000, 0.000000, 0.000000, 0.001465, 0.003800, 0.006993,
    0.013266, 0.025400, 0.046494, 0.080917, 0.133356, 0.208246, 0.302287,
    0.403247, 0.498395, 0.575000, 0.622843, 0.634095, 0.549647, 0.416874,
    0.285337, 0.175278, 0.095138, 0.043228, 0.017411, 0.012024, 0.013102,
    0.013077, 0.012081, 0.010578, 0.008983, 0.007425, 0.005931, 0.004532,
    0.003257, 0.002131, 0.000318, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000056, 0.000000, 0.000000,
    0.000000, 0.000941, 0.002600, 0.005345, 0.011717, 0.024800, 0.047712,
    0.083701, 0.136045, 0.207822, 0.295955, 0.390253, 0.480131, 0.555000,
    0.606046, 0.631507, 0.556832, 0.424147, 0.289163, 0.177141, 0.097841,
    0.047282, 0.020878, 0.012158, 0.010256, 0.009153, 0.008373, 0.007682,
    0.006879, 0.005959, 0.004984, 0.004016, 0.003117, 0.002347, 0.001224,
    0.000549, 0.000190, 0.000039, 0.000000, 0.000000, 0.000000, 0.000001,
    0.000002, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000022, 0.000000, 0.000000, 0.000000, 0.000416, 0.001400,
    0.003697, 0.010167, 0.024200, 0.048931, 0.086485, 0.138734, 0.207399,
    0.289623, 0.377259, 0.461866, 0.535000, 0.589248, 0.628919, 0.564016,
    0.431420, 0.292989, 0.179004, 0.100544, 0.051335, 0.024346, 0.012292,
    0.007410, 0.005228, 0.004665, 0.004785, 0.004774, 0.004493, 0.004036,
    0.003500, 0.002978, 0.002563, 0.002130, 0.002127, 0.002424, 0.002805,
    0.003000, 0.002817, 0.002352, 0.001770, 0.001196, 0.000703, 0.000229,
    0.000000, 0.000007, 0.000005, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000005, 0.000000, 0.000000, 0.000000,
    0.000014, 0.000000, 0.000122, 0.001400, 0.005307, 0.014281, 0.031000,
    0.058038, 0.097544, 0.151565, 0.221976, 0.305558, 0.393201, 0.475464,
    0.542909, 0.587840, 0.609054, 0.546207, 0.421925, 0.277490, 0.159335,
    0.085161, 0.043610, 0.022070, 0.010422, 0.004700, 0.002645, 0.002793,
    0.003712, 0.004166, 0.003999, 0.003436, 0.002697, 0.002007, 0.001576,
    0.001492, 0.001769, 0.001660, 0.001260, 0.001209, 0.001893, 0.002741,
    0.002973, 0.002363, 0.001421, 0.000411, 0.000000, 0.000029, 0.000021,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000018, 0.000000, 0.000000, 0.000000, 0.000055, 0.000000, 0.000000,
    0.001400, 0.006154, 0.016642, 0.035400, 0.064715, 0.105862, 0.159866,
    0.227612, 0.305796, 0.386268, 0.460610, 0.520400, 0.558982, 0.575832,
    0.526147, 0.417318, 0.275093, 0.156348, 0.084119, 0.045753, 0.027053,
    0.016749, 0.010796, 0.007303, 0.005456, 0.004487, 0.003722, 0.003003,
    0.002352, 0.001790, 0.001340, 0.001021, 0.000714, 0.000626, 0.000511,
    0.000334, 0.000198, 0.000172, 0.000197, 0.000187, 0.000112, 0.000019,
    0.000000, 0.000000, 0.000017, 0.000012, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000004, 0.000000, 0.000000,
    0.000000, 0.000038, 0.000200, 0.000777, 0.002800, 0.007911, 0.019454,
    0.041200, 0.076049, 0.123422, 0.181874, 0.249886, 0.323855, 0.397764,
    0.465463, 0.520800, 0.558416, 0.572514, 0.497886, 0.372851, 0.252579,
    0.157277, 0.092633, 0.051852, 0.027596, 0.013707, 0.006931, 0.004900,
    0.005699, 0.007432, 0.008458, 0.008554, 0.007993, 0.007045, 0.005985,
    0.005070, 0.003878, 0.003191, 0.002606, 0.001990, 0.001384, 0.000832,
    0.000390, 0.000116, 0.000010, 0.000000, 0.000007, 0.000004, 0.000000,
    0.000000, 0.000001, 0.000002, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000006, 0.000000, 0.000000, 0.000000, 0.000019, 0.000000,
    0.000156, 0.001800, 0.006744, 0.017699, 0.037600, 0.068973, 0.112708,
    0.169284, 0.239045, 0.318163, 0.397990, 0.469610, 0.524105, 0.554722,
    0.554065, 0.497074, 0.396279, 0.269397, 0.161083, 0.090162, 0.048017,
    0.024842, 0.012401, 0.006361, 0.004095, 0.004100, 0.004923, 0.005305,
    0.005078, 0.004452, 0.003641, 0.002854, 0.002293, 0.001879, 0.001846,
    0.001555, 0.000960, 0.000387, 0.000089, 0.000004, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000029, 0.000000, 0.000000, 0.000000, 0.000623, 0.002400, 0.006543,
    0.016388, 0.035800, 0.067848, 0.112411, 0.168569, 0.235317, 0.308997,
    0.382884, 0.450084, 0.503701, 0.538156, 0.549611, 0.491707, 0.387148,
    0.269874, 0.170214, 0.100537, 0.055446, 0.028772, 0.014620, 0.007837,
    0.005105, 0.004838, 0.005530, 0.005866, 0.005639, 0.005022, 0.004189,
    0.003314, 0.002564, 0.001577, 0.001091, 0.000883, 0.000814, 0.000800,
    0.000778, 0.000775, 0.000832, 0.000936, 0.001001, 0.000867, 0.000267,
    0.000000, 0.000000, 0.000097, 0.000135, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000002, 0.000000, 0.000000, 0.000000, 0.000005,
    0.000000, 0.000355, 0.003000, 0.010212, 0.024117, 0.046800, 0.079906,
    0.123315, 0.176467, 0.238726, 0.307137, 0.376073, 0.439753, 0.492400,
    0.529086, 0.549353, 0.495586, 0.391295, 0.276305, 0.177487, 0.106079,
    0.058438, 0.030275, 0.016424, 0.009721, 0.005928, 0.004057, 0.003259,
    0.002785, 0.002433, 0.002191, 0.002051, 0.001999, 0.002025, 0.002167,
    0.002122, 0.001560, 0.000674, 0.000000, 0.000000, 0.000076, 0.000234,
    0.000179, 0.000034, 0.000000, 0.000000, 0.000030, 0.000021, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000001, 0.000000, 0.000022, 0.000200, 0.000934, 0.003600,
    0.010024, 0.022867, 0.045000, 0.078759, 0.124347, 0.181436, 0.249587,
    0.325060, 0.400299, 0.467535, 0.519000, 0.548563, 0.545304, 0.473373,
    0.367412, 0.262308, 0.172890, 0.105220, 0.057732, 0.028524, 0.014183,
    0.007731, 0.004727, 0.003822, 0.003801, 0.003611, 0.003087, 0.002377,
    0.001628, 0.000988, 0.000594, 0.000498, 0.000755, 0.000728, 0.000366,
    0.000000, 0.000000, 0.000000, 0.000058, 0.000210, 0.000363, 0.000574,
    0.000794, 0.000719, 0.000388, 0.000037, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000039, 0.000000,
    0.000000, 0.000000, 0.001033, 0.004801, 0.013438, 0.029206, 0.054400,
    0.090750, 0.137734, 0.194266, 0.259184, 0.328933, 0.397198, 0.457510,
    0.503400, 0.529722, 0.525595, 0.456528, 0.355518, 0.256389, 0.170420,
    0.102879, 0.054322, 0.025090, 0.012720, 0.008074, 0.005628, 0.004507,
    0.004041, 0.003639, 0.003182, 0.002714, 0.002276, 0.001913, 0.001664,
    0.001465, 0.001427, 0.001284, 0.001023, 0.000795, 0.000704, 0.000660,
    0.000534, 0.000298, 0.000050, 0.000000, 0.000000, 0.000044, 0.000031,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000028, 0.000000, 0.000000, 0.000000, 0.000110, 0.000200, 0.000812,
    0.005200, 0.016692, 0.036195, 0.064013, 0.100443, 0.145767, 0.200261,
    0.264083, 0.333777, 0.401706, 0.460003, 0.500800, 0.518381, 0.497632,
    0.434995, 0.347443, 0.245419, 0.156649, 0.093558, 0.052209, 0.027952,
    0.015782, 0.009923, 0.006910, 0.005715, 0.005418, 0.005211, 0.004893,
    0.004481, 0.003989, 0.003431, 0.002825, 0.001619, 0.000748, 0.000560,
    0.000882, 0.001204, 0.001148, 0.000827, 0.000470, 0.000205, 0.000031,
    0.000000, 0.000000, 0.000026, 0.000019, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000054,
    0.000000, 0.000000, 0.000000, 0.001124, 0.004200, 0.010859, 0.024931,
    0.050800, 0.091387, 0.143780, 0.203607, 0.266506, 0.328372, 0.385401,
    0.433806, 0.469800, 0.490392, 0.486984, 0.431273, 0.342620, 0.246502,
    0.162732, 0.100142, 0.056908, 0.030736, 0.017953, 0.011639, 0.007719,
    0.005463, 0.004289, 0.003667, 0.003372, 0.003284, 0.003283, 0.003249,
    0.003068, 0.002275, 0.001290, 0.000551, 0.000159, 0.000000, 0.000000,
    0.000000, 0.000007, 0.000010, 0.000002, 0.000000, 0.000000, 0.000002,
    0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000008, 0.000000, 0.000050, 0.000600,
    0.002443, 0.007401, 0.017713, 0.036239, 0.066000, 0.108896, 0.162360,
    0.222707, 0.286232, 0.348561, 0.404555, 0.449029, 0.476800, 0.484255,
    0.450492, 0.379651, 0.294009, 0.207352, 0.135774, 0.085312, 0.051772,
    0.030514, 0.017409, 0.009614, 0.005251, 0.003221, 0.002454, 0.002011,
    0.001653, 0.001390, 0.001231, 0.001185, 0.001259, 0.001610, 0.001836,
    0.001509, 0.000828, 0.000397, 0.000620, 0.001149, 0.001473, 0.001411,
    0.001230, 0.001255, 0.001833, 0.001897, 0.001107, 0.000149, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000023, 0.000000,
    0.000000, 0.000000, 0.000095, 0.000200, 0.001118, 0.007201, 0.023066,
    0.050837, 0.092018, 0.146811, 0.210198, 0.275862, 0.337537, 0.390554,
    0.432095, 0.459443, 0.469882, 0.461802, 0.402442, 0.311653, 0.221359,
    0.149619, 0.097243, 0.061974, 0.039450, 0.025216, 0.015684, 0.009390,
    0.005447, 0.003220, 0.002086, 0.001484, 0.001194, 0.001116, 0.001151,
    0.001199, 0.001166, 0.000848, 0.000476, 0.000366, 0.000490, 0.000601,
    0.000518, 0.000320, 0.000140, 0.000045, 0.000005, 0.000000, 0.000000,
    0.000004, 0.000003, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000};

  std::array<double, 960> constexpr lambda1_arr = {
    0.000000, 0.000000, 0.000000, 0.000005, 0.000000, 0.000000, 0.000000,
    0.000050, 0.000000, 0.000000, 0.000000, 0.000000, 0.001210, 0.003587,
    0.006758, 0.010590, 0.016091, 0.025590, 0.041492, 0.066200, 0.101608,
    0.203457, 0.341122, 0.480704, 0.588506, 0.662430, 0.696216, 0.677593,
    0.594091, 0.453524, 0.312011, 0.200446, 0.117021, 0.058505, 0.021512,
    0.001792, 0.000000, 0.000000, 0.000743, 0.004785, 0.006568, 0.003861,
    0.001245, 0.000112, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000057, 0.000041, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000005, 0.000000, 0.000000, 0.000000, 0.000050, 0.000000, 0.000000,
    0.000000, 0.000000, 0.001210, 0.003587, 0.006758, 0.010590, 0.016091,
    0.025590, 0.041492, 0.066200, 0.101608, 0.203457, 0.341122, 0.480704,
    0.588506, 0.662430, 0.696216, 0.677593, 0.594091, 0.453524, 0.312011,
    0.200446, 0.117021, 0.058505, 0.021512, 0.001792, 0.000000, 0.000000,
    0.000743, 0.004785, 0.006568, 0.003861, 0.001245, 0.000112, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000057, 0.000041, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000002, 0.000000, 0.000000,
    0.000000, 0.000025, 0.000000, 0.000000, 0.000000, 0.000200, 0.001113,
    0.003022, 0.006364, 0.011589, 0.019610, 0.031870, 0.049842, 0.075000,
    0.108602, 0.202519, 0.332044, 0.469791, 0.580418, 0.651067, 0.674694,
    0.648761, 0.570959, 0.451102, 0.327820, 0.223451, 0.139566, 0.076835,
    0.035320, 0.011678, 0.001383, 0.000000, 0.002720, 0.005488, 0.006411,
    0.004076, 0.001900, 0.000911, 0.000596, 0.000495, 0.000445, 0.000354,
    0.000196, 0.000033, 0.000000, 0.000000, 0.000029, 0.000020, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000002,
    0.000000, 0.000000, 0.000000, 0.000007, 0.000000, 0.000000, 0.000000,
    0.000134, 0.000351, 0.000600, 0.001015, 0.002457, 0.005970, 0.012589,
    0.023130, 0.038150, 0.058192, 0.083800, 0.115597, 0.201581, 0.322966,
    0.458878, 0.572330, 0.639703, 0.653173, 0.619929, 0.547826, 0.448679,
    0.343630, 0.246456, 0.162111, 0.095166, 0.049129, 0.021565, 0.007974,
    0.003853, 0.004698, 0.006190, 0.006255, 0.004292, 0.002556, 0.001711,
    0.001395, 0.001282, 0.001235, 0.001164, 0.001027, 0.000848, 0.000559,
    0.000138, 0.000000, 0.000000, 0.000048, 0.000067, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000000, 0.000000,
    0.000003, 0.000000, 0.000000, 0.000000, 0.000108, 0.000502, 0.001400,
    0.003076, 0.006020, 0.010775, 0.017896, 0.028236, 0.042991, 0.063380,
    0.090618, 0.125623, 0.217498, 0.334388, 0.465426, 0.584314, 0.654230,
    0.662326, 0.620360, 0.541164, 0.439401, 0.333905, 0.236870, 0.153181,
    0.087404, 0.043018, 0.017407, 0.005838, 0.003576, 0.005886, 0.008230,
    0.008397, 0.005672, 0.003734, 0.003348, 0.003397, 0.003014, 0.002361,
    0.001831, 0.001579, 0.001439, 0.001106, 0.000313, 0.000000, 0.000000,
    0.000112, 0.000157, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000006, 0.000000, 0.000000, 0.000000, 0.000065,
    0.000000, 0.000000, 0.000000, 0.000400, 0.002459, 0.006020, 0.011191,
    0.018113, 0.027854, 0.042551, 0.064400, 0.095600, 0.137346, 0.240253,
    0.345035, 0.452874, 0.560106, 0.623706, 0.627366, 0.583829, 0.507103,
    0.411870, 0.314197, 0.224427, 0.146926, 0.085802, 0.044178, 0.019675,
    0.008000, 0.004857, 0.005953, 0.007173, 0.005829, 0.002381, 0.000264,
    0.000083, 0.000613, 0.000894, 0.001033, 0.001379, 0.002015, 0.002669,
    0.003149, 0.002715, 0.001648, 0.000614, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000002,
    0.000000, 0.000000, 0.000000, 0.000022, 0.000000, 0.000000, 0.000019,
    0.000600, 0.002064, 0.005215, 0.010958, 0.020196, 0.033866, 0.052941,
    0.078394, 0.111200, 0.151922, 0.252407, 0.368175, 0.482429, 0.573762,
    0.621328, 0.618858, 0.574659, 0.497681, 0.400173, 0.302070, 0.215247,
    0.142303, 0.085431, 0.046143, 0.022111, 0.009673, 0.005166, 0.004928,
    0.005443, 0.004750, 0.003192, 0.002489, 0.002873, 0.003619, 0.004097,
    0.004092, 0.003490, 0.002405, 0.001249, 0.000159, 0.000000, 0.000080,
    0.000057, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000001, 0.000000, 0.000000, 0.000000, 0.000002, 0.000000,
    0.000000, 0.000000, 0.000077, 0.000454, 0.001400, 0.003210, 0.006287,
    0.011060, 0.017981, 0.028161, 0.043480, 0.065859, 0.097219, 0.138647,
    0.241855, 0.352365, 0.462864, 0.563583, 0.623500, 0.629453, 0.587404,
    0.504173, 0.393940, 0.288156, 0.203088, 0.137102, 0.087788, 0.052720,
    0.029393, 0.015272, 0.007823, 0.004515, 0.002903, 0.001274, 0.000964,
    0.001118, 0.001267, 0.001193, 0.000777, 0.000247, 0.000000, 0.000000,
    0.000000, 0.000081, 0.000046, 0.000000, 0.000000, 0.000018, 0.000025,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001, 0.000000,
    0.000000, 0.000000, 0.000004, 0.000000, 0.000000, 0.000000, 0.000118,
    0.000518, 0.001400, 0.003059, 0.006156, 0.011443, 0.019679, 0.031835,
    0.049126, 0.072778, 0.104021, 0.143584, 0.242131, 0.354712, 0.463917,
    0.551613, 0.602816, 0.610206, 0.573965, 0.494630, 0.383453, 0.277004,
    0.194109, 0.131899, 0.086562, 0.054460, 0.032941, 0.019689, 0.012392,
    0.008736, 0.006496, 0.003298, 0.001636, 0.001339, 0.001814, 0.002202,
    0.001890, 0.001191, 0.000627, 0.000421, 0.000400, 0.000350, 0.000109,
    0.000000, 0.000000, 0.000039, 0.000055, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000001, 0.000000, 0.000000, 0.000000, 0.000003,
    0.000000, 0.000000, 0.000000, 0.000074, 0.000306, 0.000800, 0.001837,
    0.004407, 0.009677, 0.018804, 0.032693, 0.051958, 0.077195, 0.109000,
    0.147733, 0.243365, 0.356971, 0.465546, 0.544325, 0.584691, 0.584574,
    0.547533, 0.477383, 0.383620, 0.289140, 0.207198, 0.139314, 0.086495,
    0.049274, 0.025516, 0.012158, 0.006136, 0.004387, 0.003963, 0.003233,
    0.002592, 0.002052, 0.001597, 0.001189, 0.000811, 0.000513, 0.000362,
    0.000354, 0.000394, 0.000364, 0.000117, 0.000000, 0.000000, 0.000043,
    0.000060, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000004, 0.000000, 0.000000, 0.000000, 0.000047, 0.000000,
    0.000000, 0.000000, 0.000800, 0.003041, 0.007137, 0.013600, 0.022951,
    0.036094, 0.054371, 0.079151, 0.111800, 0.153135, 0.255771, 0.371655,
    0.474644, 0.543932, 0.576878, 0.573503, 0.535452, 0.464465, 0.369026,
    0.273594, 0.192202, 0.125965, 0.075435, 0.040722, 0.019456, 0.008404,
    0.004333, 0.004009, 0.004330, 0.003376, 0.001535, 0.000300, 0.000000,
    0.000002, 0.000035, 0.000016, 0.000000, 0.000000, 0.000000, 0.000006,
    0.000003, 0.000000, 0.000000, 0.000001, 0.000002, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000003, 0.000000,
    0.000000, 0.000000, 0.000036, 0.000000, 0.000000, 0.000095, 0.001200,
    0.003741, 0.008302, 0.015472, 0.025854, 0.040411, 0.060520, 0.087583,
    0.123000, 0.167473, 0.274502, 0.386978, 0.482249, 0.546567, 0.577150,
    0.572668, 0.532633, 0.456624, 0.353666, 0.255138, 0.177391, 0.118264,
    0.074793, 0.044095, 0.023765, 0.011558, 0.005232, 0.002545, 0.001336,
    0.000236, 0.000148, 0.000285, 0.000358, 0.000401, 0.000438, 0.000438,
    0.000356, 0.000198, 0.000034, 0.000000, 0.000000, 0.000029, 0.000021,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000004, 0.000000, 0.000000, 0.000000, 0.000045,
    0.000000, 0.000000, 0.000092, 0.001400, 0.004439, 0.009947, 0.018678,
    0.031392, 0.048960, 0.072379, 0.102657, 0.140800, 0.187092, 0.293012,
    0.395200, 0.481653, 0.545367, 0.569833, 0.550802, 0.496875, 0.417219,
    0.324092, 0.236856, 0.165455, 0.109278, 0.067255, 0.038148, 0.019781,
    0.009655, 0.005265, 0.004112, 0.003792, 0.002713, 0.001515, 0.000791,
    0.000588, 0.000600, 0.000576, 0.000483, 0.000340, 0.000174, 0.000028,
    0.000000, 0.000000, 0.000025, 0.000018, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000002,
    0.000000, 0.000000, 0.000000, 0.000022, 0.000000, 0.000000, 0.000138,
    0.001000, 0.003035, 0.007250, 0.014776, 0.026738, 0.044051, 0.067391,
    0.097420, 0.134800, 0.179656, 0.283180, 0.388018, 0.472647, 0.524628,
    0.543574, 0.530680, 0.488056, 0.417865, 0.328444, 0.242692, 0.172743,
    0.117690, 0.076061, 0.046307, 0.026459, 0.014402, 0.008018, 0.005194,
    0.003886, 0.002739, 0.002492, 0.002020, 0.001209, 0.000590, 0.000529,
    0.000734, 0.000768, 0.000476, 0.000084, 0.000000, 0.000000, 0.000073,
    0.000052, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000000, 0.000000,
    0.000008, 0.000000, 0.000037, 0.000535, 0.002000, 0.005075, 0.010953,
    0.020966, 0.036434, 0.058316, 0.087157, 0.123478, 0.167800, 0.219926,
    0.334592, 0.438206, 0.508377, 0.539125, 0.535295, 0.501774, 0.442905,
    0.363022, 0.271840, 0.191650, 0.132568, 0.090554, 0.060949, 0.039506,
    0.024273, 0.014093, 0.007814, 0.004281, 0.002372, 0.001228, 0.001803,
    0.002052, 0.001521, 0.000782, 0.000300, 0.000068, 0.000000, 0.000000,
    0.000000, 0.000142, 0.000496, 0.000598, 0.000365, 0.000049, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000012, 0.000000, 0.000000, 0.000000, 0.000152, 0.000200, 0.000090,
    0.000714, 0.003201, 0.008754, 0.018873, 0.035134, 0.059073, 0.091135,
    0.130495, 0.176260, 0.227537, 0.282994, 0.393732, 0.481227, 0.526511,
    0.525991, 0.484638, 0.413462, 0.329069, 0.248291, 0.182664, 0.130776,
    0.090199, 0.059440, 0.037049, 0.021589, 0.011729, 0.006169, 0.003610,
    0.002753, 0.002351, 0.001408, 0.000581, 0.000231, 0.000288, 0.000401,
    0.000309, 0.000106, 0.000000, 0.000000, 0.000000, 0.000035, 0.000020,
    0.000000, 0.000000, 0.000008, 0.000011, 0.000000, 0.000000, 0.000000,
    0.000000};

  std::array<double, 960> constexpr lambda2_arr = {
    0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000000, 0.000000,
    0.000008, 0.000000, 0.000000, 0.000000, 0.000000, 0.000148, 0.000265,
    0.000118, 0.000000, 0.000000, 0.000000, 0.000000, 0.000400, 0.004824,
    0.016421, 0.024494, 0.037917, 0.067747, 0.107650, 0.157619, 0.225006,
    0.317442, 0.429533, 0.524837, 0.582537, 0.602247, 0.584468, 0.531018,
    0.451083, 0.356413, 0.258759, 0.169872, 0.100884, 0.024982, 0.001988,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000399, 0.000384,
    0.000264, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000001, 0.000000, 0.000000, 0.000000, 0.000008, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000148, 0.000265, 0.000118, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000400, 0.004824, 0.016421, 0.024494, 0.037917,
    0.067747, 0.107650, 0.157619, 0.225006, 0.317442, 0.429533, 0.524837,
    0.582537, 0.602247, 0.584468, 0.531018, 0.451083, 0.356413, 0.258759,
    0.169872, 0.100884, 0.024982, 0.001988, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000399, 0.000384, 0.000264, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000004, 0.000000, 0.000000, 0.000000, 0.000000, 0.000074,
    0.000132, 0.000094, 0.000000, 0.000000, 0.000000, 0.000135, 0.001600,
    0.004233, 0.012830, 0.025386, 0.044951, 0.075431, 0.117559, 0.172217,
    0.240617, 0.323966, 0.416521, 0.495978, 0.548432, 0.570933, 0.561011,
    0.517720, 0.448654, 0.364371, 0.275438, 0.192417, 0.125348, 0.043469,
    0.009935, 0.000000, 0.000000, 0.001455, 0.002188, 0.001414, 0.000519,
    0.000192, 0.000187, 0.000212, 0.000075, 0.000000, 0.000000, 0.000028,
    0.000039, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000000, 0.000000,
    0.000009, 0.000015, 0.000000, 0.000000, 0.000000, 0.000070, 0.000350,
    0.000817, 0.001419, 0.002099, 0.002800, 0.003643, 0.009238, 0.026278,
    0.051986, 0.083114, 0.127468, 0.186815, 0.256229, 0.330490, 0.403509,
    0.467119, 0.514326, 0.539619, 0.537555, 0.504423, 0.446224, 0.372330,
    0.292117, 0.214962, 0.149811, 0.061957, 0.017883, 0.002083, 0.001764,
    0.005876, 0.005986, 0.003167, 0.000639, 0.000000, 0.000110, 0.000435,
    0.000201, 0.000000, 0.000000, 0.000076, 0.000107, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000002, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000046, 0.000110, 0.000173, 0.000219, 0.000329, 0.000697, 0.001521,
    0.003001, 0.005362, 0.014100, 0.030520, 0.055537, 0.089652, 0.134967,
    0.191615, 0.257655, 0.331050, 0.406528, 0.471122, 0.515877, 0.537041,
    0.531088, 0.495939, 0.437598, 0.364881, 0.286607, 0.211594, 0.148254,
    0.062486, 0.018831, 0.002278, 0.000751, 0.004067, 0.004380, 0.002320,
    0.000437, 0.000000, 0.000124, 0.000394, 0.000178, 0.000000, 0.000000,
    0.000067, 0.000094, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000006, 0.000009, 0.000000, 0.000000, 0.000000, 0.000091,
    0.000325, 0.000792, 0.001631, 0.002986, 0.005000, 0.007845, 0.017225,
    0.033768, 0.062461, 0.105565, 0.159461, 0.221021, 0.287952, 0.357985,
    0.425749, 0.478554, 0.508918, 0.515717, 0.498101, 0.456228, 0.395881,
    0.324801, 0.250730, 0.181410, 0.124205, 0.049895, 0.014683, 0.001538,
    0.000000, 0.001653, 0.002492, 0.002168, 0.002067, 0.002795, 0.003917,
    0.005396, 0.006611, 0.006181, 0.004554, 0.002721, 0.001694, 0.001350,
    0.000451, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000006, 0.000009,
    0.000000, 0.000000, 0.000025, 0.000208, 0.000611, 0.001269, 0.002167,
    0.003284, 0.004600, 0.006295, 0.014922, 0.037883, 0.073505, 0.116063,
    0.167093, 0.226206, 0.290661, 0.357602, 0.421162, 0.468392, 0.492361,
    0.493256, 0.471580, 0.428587, 0.369763, 0.302063, 0.232443, 0.167859,
    0.114924, 0.046913, 0.015368, 0.003730, 0.001791, 0.003441, 0.003932,
    0.003282, 0.002591, 0.002380, 0.002396, 0.002287, 0.001616, 0.000816,
    0.000208, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000002, 0.000000, 0.000000, 0.000000, 0.000000, 0.000048, 0.000113,
    0.000174, 0.000217, 0.000361, 0.000882, 0.002067, 0.004201, 0.007583,
    0.019518, 0.040675, 0.071163, 0.110312, 0.159541, 0.218174, 0.283312,
    0.351948, 0.417703, 0.466284, 0.490341, 0.490710, 0.468581, 0.425792,
    0.367774, 0.301214, 0.232799, 0.169214, 0.116816, 0.048479, 0.015518,
    0.002287, 0.000000, 0.000031, 0.000512, 0.000236, 0.000000, 0.000000,
    0.000000, 0.000082, 0.000047, 0.000000, 0.000000, 0.000018, 0.000025,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000001, 0.000000, 0.000000, 0.000000, 0.000010, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000200, 0.000384, 0.000366, 0.000000, 0.000000,
    0.000000, 0.000164, 0.002801, 0.007577, 0.021620, 0.038048, 0.067546,
    0.116539, 0.169432, 0.220667, 0.276015, 0.341713, 0.414406, 0.467966,
    0.489765, 0.484205, 0.456433, 0.411622, 0.355101, 0.292254, 0.228466,
    0.169121, 0.119343, 0.051307, 0.015809, 0.001447, 0.000000, 0.001252,
    0.001867, 0.001022, 0.000129, 0.000000, 0.000000, 0.000105, 0.000059,
    0.000000, 0.000000, 0.000023, 0.000032, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000005, 0.000007, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000037, 0.000275, 0.000921, 0.002216, 0.004400,
    0.007741, 0.019436, 0.040508, 0.073907, 0.119436, 0.173483, 0.232509,
    0.293240, 0.352413, 0.405594, 0.445622, 0.467914, 0.470895, 0.453119,
    0.414118, 0.358885, 0.294311, 0.227288, 0.164710, 0.113132, 0.046178,
    0.014367, 0.002161, 0.000000, 0.001032, 0.001255, 0.000475, 0.000000,
    0.000000, 0.000128, 0.000380, 0.000170, 0.000000, 0.000000, 0.000064,
    0.000090, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000003, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000072, 0.000156, 0.000205, 0.000177,
    0.000171, 0.000456, 0.001307, 0.003000, 0.005901, 0.018292, 0.044507,
    0.082702, 0.128299, 0.182473, 0.243589, 0.306858, 0.367348, 0.418733,
    0.451537, 0.461827, 0.451724, 0.423615, 0.380210, 0.326033, 0.266239,
    0.205986, 0.150429, 0.104466, 0.043440, 0.013106, 0.000990, 0.000000,
    0.000039, 0.000649, 0.000299, 0.000000, 0.000000, 0.000000, 0.000105,
    0.000059, 0.000000, 0.000000, 0.000023, 0.000032, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000005, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000105, 0.000215, 0.000247, 0.000124, 0.000000, 0.000249, 0.001313,
    0.003600, 0.007520, 0.021833, 0.047318, 0.085574, 0.135024, 0.191977,
    0.252673, 0.313398, 0.370434, 0.418574, 0.449197, 0.458561, 0.448354,
    0.420501, 0.377313, 0.323283, 0.263658, 0.203686, 0.148616, 0.103429,
    0.044486, 0.015854, 0.003864, 0.000289, 0.000412, 0.000495, 0.000191,
    0.000000, 0.000000, 0.000000, 0.000065, 0.000037, 0.000000, 0.000000,
    0.000014, 0.000020, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000005,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000101, 0.000208, 0.000242,
    0.000132, 0.000078, 0.000587, 0.002185, 0.005400, 0.010686, 0.028085,
    0.055255, 0.092878, 0.141462, 0.201251, 0.268160, 0.333780, 0.389518,
    0.428078, 0.445493, 0.441682, 0.419971, 0.383831, 0.336822, 0.282986,
    0.226532, 0.171670, 0.122608, 0.083326, 0.034333, 0.012568, 0.003792,
    0.000644, 0.000199, 0.000129, 0.000041, 0.000000, 0.000000, 0.000000,
    0.000013, 0.000008, 0.000000, 0.000000, 0.000003, 0.000004, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000000,
    0.000000, 0.000003, 0.000000, 0.000000, 0.000000, 0.000059, 0.000139,
    0.000200, 0.000231, 0.000342, 0.000672, 0.001358, 0.002539, 0.004349,
    0.006924, 0.010400, 0.015002, 0.030031, 0.057228, 0.098905, 0.152800,
    0.214577, 0.278391, 0.336985, 0.383055, 0.412049, 0.426099, 0.426649,
    0.413436, 0.386120, 0.344992, 0.293847, 0.237700, 0.181566, 0.130460,
    0.089138, 0.037062, 0.013497, 0.004097, 0.001103, 0.001006, 0.001024,
    0.000789, 0.000484, 0.000225, 0.000035, 0.000000, 0.000000, 0.000030,
    0.000022, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001, 0.000000,
    0.000000, 0.000000, 0.000021, 0.000033, 0.000000, 0.000000, 0.000000,
    0.000127, 0.000719, 0.001822, 0.003558, 0.006044, 0.009400, 0.013949,
    0.031410, 0.068317, 0.119172, 0.174360, 0.236999, 0.304006, 0.365322,
    0.410600, 0.432589, 0.431716, 0.411380, 0.376803, 0.333288, 0.285773,
    0.237163, 0.189652, 0.145435, 0.106707, 0.075529, 0.035029, 0.015081,
    0.005677, 0.001483, 0.000183, 0.000000, 0.000000, 0.000033, 0.000047,
    0.000010, 0.000000, 0.000000, 0.000009, 0.000006, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001,
    0.000002, 0.000000, 0.000016, 0.000167, 0.000593, 0.001438, 0.002969,
    0.005596, 0.009737, 0.015809, 0.024378, 0.053772, 0.107250, 0.173398,
    0.238172, 0.307873, 0.377316, 0.428615, 0.443367, 0.416167, 0.372814,
    0.328982, 0.285429, 0.242263, 0.199735, 0.158896, 0.121073, 0.087597,
    0.059794, 0.038890, 0.015608, 0.007739, 0.004618, 0.002168, 0.000569,
    0.000000, 0.000000, 0.000036, 0.000052, 0.000011, 0.000000, 0.000000,
    0.000010, 0.000007, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000};

  std::array<double, 960> constexpr lambda3_arr = {
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000001, 0.000002, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000010, 0.000048, 0.000119, 0.000233, 0.000400, 0.000535,
    0.000647, 0.000000, 0.001232, 0.007765, 0.012791, 0.016135, 0.024214,
    0.043723, 0.078418, 0.124971, 0.181899, 0.250644, 0.332779, 0.428844,
    0.533614, 0.639857, 0.740343, 0.827838, 0.895538, 0.973436, 1.000000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.945102, 0.600790,
    0.139850, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001,
    0.000002, 0.000000, 0.000000, 0.000000, 0.000000, 0.000010, 0.000048,
    0.000119, 0.000233, 0.000400, 0.000535, 0.000647, 0.000000, 0.001232,
    0.007765, 0.012791, 0.016135, 0.024214, 0.043723, 0.078418, 0.124971,
    0.181899, 0.250644, 0.332779, 0.428844, 0.533614, 0.639857, 0.740343,
    0.827838, 0.895538, 0.973436, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 0.945102, 0.600790, 0.139850, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000001, 0.000002, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000009, 0.000040, 0.000083, 0.000137, 0.000200,
    0.000267, 0.000378, 0.000501, 0.003181, 0.009371, 0.015435, 0.022132,
    0.034885, 0.059314, 0.097996, 0.146208, 0.201883, 0.266726, 0.342606,
    0.430463, 0.526036, 0.623253, 0.716044, 0.798338, 0.864400, 0.947926,
    0.985290, 0.998371, 1.000000, 0.997948, 0.997329, 0.998179, 0.999067,
    0.999393, 0.999670, 1.000000, 1.000000, 0.988643, 0.938831, 0.812169,
    0.568757, 0.266600, 0.018200, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000001, 0.000002, 0.000000, 0.000000, 0.000000, 0.000000, 0.000008,
    0.000031, 0.000047, 0.000042, 0.000000, 0.000000, 0.000109, 0.001519,
    0.005130, 0.010978, 0.018079, 0.028128, 0.045556, 0.074906, 0.117573,
    0.167445, 0.221868, 0.282808, 0.352434, 0.432083, 0.518457, 0.606649,
    0.691746, 0.768838, 0.833263, 0.922417, 0.969425, 0.987369, 0.988632,
    0.985145, 0.985992, 0.989813, 0.992891, 0.993384, 0.991958, 0.987350,
    0.964678, 0.905764, 0.806551, 0.679236, 0.536723, 0.393350, 0.252450,
    0.111550, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000006, 0.000011, 0.000008, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000126, 0.000934, 0.003102, 0.007462, 0.014417, 0.023807,
    0.036880, 0.056331, 0.084912, 0.124104, 0.172325, 0.228563, 0.292847,
    0.365251, 0.445353, 0.529965, 0.614937, 0.696118, 0.769356, 0.830715,
    0.916617, 0.963174, 0.982214, 0.985160, 0.983177, 0.984752, 0.988541,
    0.990974, 0.990059, 0.985911, 0.973779, 0.930808, 0.845818, 0.714074,
    0.541851, 0.335907, 0.119850, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000024, 0.000042, 0.000030,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000200, 0.000781, 0.002635,
    0.005129, 0.009599, 0.017094, 0.026487, 0.038869, 0.057672, 0.086433,
    0.127738, 0.181805, 0.247376, 0.322068, 0.403450, 0.488977, 0.575465,
    0.659507, 0.737697, 0.806627, 0.863090, 0.939261, 0.978327, 0.994407,
    0.997849, 0.996571, 0.996520, 0.996543, 0.993899, 0.987672, 0.979392,
    0.965733, 0.929853, 0.855351, 0.734229, 0.576287, 0.392095, 0.202023,
    0.019153, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000020, 0.000037, 0.000026, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000349, 0.001253, 0.001938, 0.006171, 0.016449,
    0.028894, 0.043821, 0.066118, 0.100865, 0.150329, 0.209987, 0.276582,
    0.349165, 0.426884, 0.508590, 0.591457, 0.672078, 0.747045, 0.812950,
    0.866585, 0.937723, 0.972676, 0.985652, 0.986905, 0.984137, 0.983184,
    0.983175, 0.981735, 0.978100, 0.973657, 0.967434, 0.940573, 0.856885,
    0.708191, 0.520171, 0.319977, 0.125400, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000027, 0.000048,
    0.000034, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000462,
    0.001849, 0.003400, 0.007284, 0.014865, 0.023620, 0.035827, 0.058799,
    0.100050, 0.160666, 0.226405, 0.290498, 0.356024, 0.426488, 0.504443,
    0.587117, 0.669883, 0.748113, 0.817179, 0.872719, 0.943838, 0.977207,
    0.990997, 0.995429, 0.995591, 0.995346, 0.994346, 0.991300, 0.985873,
    0.979016, 0.967361, 0.932180, 0.852612, 0.724766, 0.569534, 0.408888,
    0.254450, 0.096550, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000001, 0.000002, 0.000001, 0.000000, 0.000000,
    0.000023, 0.000084, 0.000200, 0.000392, 0.001153, 0.002745, 0.007031,
    0.015693, 0.028575, 0.047141, 0.074586, 0.114175, 0.166760, 0.227400,
    0.293006, 0.363232, 0.437848, 0.516282, 0.596036, 0.673941, 0.746829,
    0.811530, 0.865053, 0.938565, 0.976975, 0.991840, 0.993293, 0.990549,
    0.990511, 0.991636, 0.990478, 0.985435, 0.977368, 0.961165, 0.916806,
    0.836423, 0.714948, 0.558878, 0.375213, 0.183825, 0.001035, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000004, 0.000021, 0.000055, 0.000113, 0.000200,
    0.000352, 0.001475, 0.004821, 0.010667, 0.019089, 0.031243, 0.049735,
    0.078560, 0.121762, 0.179521, 0.242798, 0.307153, 0.374170, 0.445694,
    0.522892, 0.603127, 0.682441, 0.756875, 0.822468, 0.875491, 0.944455,
    0.977515, 0.990588, 0.993590, 0.992570, 0.992135, 0.991823, 0.990017,
    0.985472, 0.977438, 0.957376, 0.898698, 0.803017, 0.669742, 0.504020,
    0.311246, 0.110350, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000002, 0.000003, 0.000000, 0.000000, 0.000000, 0.000000, 0.000013,
    0.000053, 0.000104, 0.000156, 0.000200, 0.000250, 0.000819, 0.002963,
    0.009481, 0.021227, 0.035007, 0.052229, 0.079139, 0.122181, 0.183470,
    0.254686, 0.330096, 0.408146, 0.487467, 0.566594, 0.643544, 0.716151,
    0.782250, 0.839676, 0.886404, 0.949264, 0.981517, 0.995171, 0.998319,
    0.996533, 0.994308, 0.992345, 0.990458, 0.987945, 0.983416, 0.969845,
    0.920437, 0.825136, 0.687864, 0.530897, 0.377313, 0.233650, 0.082950,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000026, 0.000046, 0.000033, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000200, 0.000812, 0.002611, 0.004695, 0.010358, 0.021886, 0.036074,
    0.054387, 0.083278, 0.129404, 0.194174, 0.266420, 0.339600, 0.413736,
    0.489132, 0.565752, 0.641648, 0.714206, 0.780811, 0.838849, 0.885868,
    0.948118, 0.978980, 0.991415, 0.994059, 0.992762, 0.992013, 0.991464,
    0.989666, 0.985731, 0.979519, 0.965339, 0.919318, 0.828528, 0.698474,
    0.559701, 0.443847, 0.348450, 0.236550, 0.124650, 0.012750, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000015, 0.000026, 0.000019,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000200, 0.000647, 0.002632,
    0.006832, 0.013981, 0.024616, 0.039351, 0.061203, 0.095597, 0.148047,
    0.219139, 0.297653, 0.376673, 0.455406, 0.533324, 0.609729, 0.682972,
    0.751072, 0.812046, 0.863914, 0.904840, 0.956380, 0.979614, 0.988635,
    0.991069, 0.990368, 0.989095, 0.986798, 0.982355, 0.975543, 0.967342,
    0.954038, 0.922240, 0.862856, 0.766901, 0.634424, 0.465869, 0.283950,
    0.115450, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000008, 0.000013,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000057, 0.000227, 0.000392,
    0.000475, 0.000400, 0.000181, 0.000913, 0.006165, 0.015838, 0.027982,
    0.042826, 0.064013, 0.098481, 0.153289, 0.228545, 0.307765, 0.382359,
    0.454195, 0.525593, 0.598274, 0.670601, 0.739766, 0.802963, 0.857385,
    0.900408, 0.954297, 0.978765, 0.989422, 0.993879, 0.994590, 0.993407,
    0.990278, 0.984720, 0.976383, 0.965099, 0.942308, 0.887327, 0.809161,
    0.711756, 0.605281, 0.500176, 0.399500, 0.295700, 0.191900, 0.088100,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000029, 0.000052,
    0.000036, 0.000000, 0.000000, 0.000000, 0.000000, 0.000200, 0.000878,
    0.003202, 0.006896, 0.017103, 0.035969, 0.056573, 0.080288, 0.117322,
    0.178241, 0.264583, 0.356275, 0.441739, 0.521221, 0.595480, 0.665067,
    0.729370, 0.787372, 0.838055, 0.880402, 0.913491, 0.954733, 0.972701,
    0.978939, 0.979188, 0.975445, 0.968964, 0.958502, 0.942283, 0.920305,
    0.894947, 0.854825, 0.775179, 0.660453, 0.510595, 0.347733, 0.194959,
    0.054626, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000001, 0.000002, 0.000002, 0.000000, 0.000048,
    0.000255, 0.000735, 0.001601, 0.002934, 0.006806, 0.012138, 0.026082,
    0.053782, 0.089750, 0.134230, 0.193826, 0.275394, 0.377552, 0.479267,
    0.568901, 0.647249, 0.715646, 0.775298, 0.826700, 0.870095, 0.905728,
    0.933844, 0.954726, 0.978658, 0.987442, 0.990928, 0.992667, 0.992150,
    0.988492, 0.979684, 0.963475, 0.938350, 0.903775, 0.833817, 0.678901,
    0.496403, 0.312320, 0.155206, 0.053726, 0.000000, 0.000000, 0.000000,
    0.000000};

  template <size_t N>
  std::vector<double>
  make_short_vec(std::array<double, N> const& a)
  {
    static_assert(N != 0, "make_short_vec requires a nonzero-length array");
    return {a.begin(), a.end() - 1};
  }

  // make_vec creates an std::vector<double> from an std::array, using all the
  // values in the std::array.
  template <size_t N>
  std::vector<double>
  make_vec(std::array<double, N> const& a)
  {
    return {a.begin(), a.end()};
  }

  // Create an Interp2D from an x-axis, y-axis, and z "matrix", with the matrix
  // unrolled into a one-dimenstional array.
template <size_t M, std::size_t N>
y3_cluster::Interp2D
make_Interp2D_aux(std::array<double, M> const& xs,
                    std::array<double, N> const& ys,
                    std::array<double, (N) * (M)> const& zs)
{
    return {make_vec(xs), make_vec(ys), make_vec(zs)};
}

auto make_Interp2D = [](auto zs) 
{
    return make_Interp2D_aux(lt_bins, zt_bins, zs);
  };

template<class T>
struct int_lc_lt_des_t {
  public:
	int_lc_lt_des_t(): lambda0_interp(lt_bins, zt_bins, lambda0_arr), 
					   lambda1_interp(lt_bins, zt_bins, lambda1_arr), 
					   lambda2_interp(lt_bins, zt_bins, lambda2_arr),
					   lambda3_interp(lt_bins, zt_bins, lambda3_arr){}
	
    typename T::Interp2D const lambda0_interp;
    typename T::Interp2D const lambda1_interp;
    typename T::Interp2D const lambda2_interp;
    typename T::Interp2D const lambda3_interp;
	
	__host__ __device__
    double
    operator()(double lc, double lt, double zt) const
    {
      double val = 0;
      if ((lc >= 20) & (lc < 30)) {
        val = lambda0_interp(lt, zt);
      } else if ((lc >= 30) & (lc < 45)) {
        val = lambda1_interp(lt, zt);
      } else if ((lc >= 45) & (lc < 60)) {
        val = lambda2_interp(lt, zt);
      } else {
        val = lambda3_interp(lt, zt);
      }
      return val;
    }
};

/*TEST_CASE("Model with member Interp2D inialized from std::array, Interp2D on a known point")
{
	double const lc = 0x1.b8p+4;
    double const lt = 0x1.b8p+4;
    double const zt = 0x1.cccccccccccccp-2;
	
	y3_cluster::INT_LC_LT_DES_t lc_lt; 
	double result = lc_lt(lc, lt, zt+.01);
	
	int_lc_lt_des_t<GPU> d_lc_lt;
	double cpu_result = d_lc_lt(lc, lt, zt+.01);
	
	int_lc_lt_des_t<GPU> *d_integrand;
    cudaMallocManaged((void**)&d_integrand, sizeof(int_lc_lt_des_t<GPU>));
    cudaDeviceSynchronize();
	memcpy(d_integrand, &d_lc_lt, sizeof(int_lc_lt_des_t<GPU>));
	
	double *gpu_result;
    cudaMallocManaged((void**)&gpu_result, sizeof(double));
		
	testKernel<<<1, 1>>>(d_integrand, lc, lt, zt+.01, gpu_result);
	cudaDeviceSynchronize();
	
	CHECK(*gpu_result == cpu_result);
	CHECK(cpu_result == result);
}*/

template<class T>
struct omega_z_des {
  public:
    omega_z_des() = default;
	
	__host__ __device__
    double
    operator()(double zt) const
    {
      const typename T::polynomial<6> SDSS_fit{{0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063}};
      const typename T::polynomial<6> SDSS_fit2{{1.33647377e+4,
                                                 1.35291046e+3,
                                                 -1.26204891e+2,
                                                 -2.83454918e+1,
                                                 -2.26465905,
                                                 3.84958753e-1}};
												  
      const typename T::polynomial<6> SDSS_fit3{{0, 0, -1.88101967, 4.8071839, -4.11424324, 1.18196785}};
    
      if (zt < 0.504) {
        return SDSS_fit(zt);
      } else if (zt < 0.7) {
        return SDSS_fit2(zt - 0.6);
      } else {
        return SDSS_fit3(zt);
      }
    }
};

/*TEST_CASE("Omega_z DES to Test quad::Polynomial")
{
	double const zt = 0x1.cccccccccccccp-2;
	y3_cluster::OMEGA_Z_DES omega_z;
	double result = omega_z(zt);
	
	omega_z_des<GPU> cpu_omega_z;
	double cpu_result = cpu_omega_z(zt);
	
	omega_z_des<GPU>* dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(omega_z_des<GPU>));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &cpu_omega_z, sizeof(omega_z_des<GPU>));
		
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
	
	testKernel<omega_z_des<GPU>><<<1,1>>>(dhmf2, zt, gpu_result);
	cudaDeviceSynchronize();
	CHECK(result == cpu_result);
	CHECK(result == *gpu_result);
} */

class sigma_photoz_des_t {
  public:
    sigma_photoz_des_t() = default;

	__host__ __device__
    double
    operator()(double zt) const
    {
      double poly_coeff[] = {-40358.8315,
                             2798.08304,
                             9333.80185,
                             -657.348248,
                             -840.565610,
                             46.8506649,
                             37.8839498,
                             -0.868811858,
                             -0.808928182,
                             0.00890199353,
                             0.0139811265};
      double _sigma = 0;
      double z_for_fit = zt;

      // We do not extrapolate outside of the data range
      if (z_for_fit < 0.15) {
        z_for_fit = 0.15;
      }
      if (z_for_fit > 0.7) {
        z_for_fit = 0.7;
      }

      // Compute the fit at pivot (z-.4)
      z_for_fit = z_for_fit - 0.4;
      for (int ii = 0; ii < 10; ii++) {
        _sigma = (poly_coeff[ii] + _sigma) * z_for_fit;
      }
      _sigma = _sigma + poly_coeff[10];

      return _sigma;
    }
  };

class int_zo_zt_des_t {
  public:
  __device__ __host__
    int_zo_zt_des_t() {}

	__device__ __host__
    double
    operator()(double zomin, double zomax, double zt) const
    {
      double _sigma = _sigma_photoz_des(zt);
      double base = sqrt(2) * _sigma;
      return (erf((zomax - zt) / base) - erf((zomin - zt) / base)) /
             2.0;
    }
	
  private:
    sigma_photoz_des_t _sigma_photoz_des;
  };

/*TEST_CASE("Simple model"){
	double const zt = 0x1.cccccccccccccp-2;
	double zo_low_ = 0.0;
	double zo_high_ = 0.0;
	
	y3_cluster::INT_ZO_ZT_DES_t int_zo_zt;
	double result = int_zo_zt(zo_low_, zo_high_, zt);
	
	int_zo_zt_des_t d_int_zo_zt;
	int_zo_zt_des_t *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(int_zo_zt_des_t));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &d_int_zo_zt, sizeof(int_zo_zt_des_t));
		
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
		
	testKernel<int_zo_zt_des_t><<<1,1>>>(dhmf2, zo_low_, zo_high_, zt, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
}*/

std::array<double, 42> test_lsat = {
      0.500000,   0.600000,   0.700000,   0.800000,   0.900000,   1.000000,
      2.000000,   3.000000,   4.000000,   5.000000,   6.000000,   7.000000,
      8.000000,   9.000000,   10.000000,  11.000000,  12.000000,  13.000000,
      14.000000,  24.000000,  34.000000,  44.000000,  54.000000,  64.000000,
      74.000000,  84.000000,  94.000000,  104.000000, 114.000000, 124.000000,
      134.000000, 144.000000, 154.000000, 164.000000, 174.000000, 184.000000,
      194.000000, 204.000000, 214.000000, 224.000000, 234.000000, 244.000000};
	
// Read from sig_intr_grid.dat
std::array<double, 14> test_sigintr = {0.050000,
                                       0.100000,
                                       0.200000,
                                       0.300000,
                                       0.400000,
                                       0.500000,
                                       0.600000,
                                       0.700000,
                                       0.800000,
                                       0.900000,
                                       1.000000,
                                       1.300000,
                                       1.700000,
                                       2.000000};

// Read from sig_skew_table.dat;
std::array<std::array<double, 42>, 14> sig_skewnorml_flat = {
      {
	   {1.211319,  1.307980,  1.455547,  1.242204,  1.260416,  1.287859,
        1.604654,  1.883786,  2.133369,  2.357737,  2.565367,  2.759611,
        2.939715,  3.113878,  3.277425,  3.435430,  3.584226,  3.730049,
        3.872571,  5.091403,  6.118126,  7.024295,  7.861908,  8.647303,
        9.392576,  10.110787, 10.799733, 11.464516, 12.127780, 12.770096,
        13.393245, 14.018971, 14.625637, 15.228590, 15.823349, 16.405359,
        16.995068, 17.565897, 18.144194, 18.704318, 19.271479, 19.830625},
		
       {1.211227,  1.309620,  1.455854,  1.241931,  1.259493,  1.287663,
        1.605865,  1.898566,  2.160331,  2.397456,  2.621481,  2.828296,
        3.026425,  3.213295,  3.393675,  3.568839,  3.735769,  3.899172,
        4.061805,  5.508045,  6.791948,  7.997309,  9.155444,  10.276979,
        11.376409, 12.460638, 13.535322, 14.584471, 15.645481, 16.685029,
        17.736828, 18.756013, 19.791557, 20.837728, 21.859120, 22.878280,
        23.911779, 24.911220, 25.953940, 26.954718, 27.983187, 28.968243},
		
       {1.211671,  1.309826,  1.456078,  1.242446,  1.260993,  1.290490,
        1.647297,  1.975678,  2.278038,  2.560308,  2.830641,  3.086770,
        3.336938,  3.585123,  3.821795,  4.058084,  4.285488,  4.513090,
        4.741029,  6.911372,  9.004724,  11.059253, 13.091911, 15.109313,
        17.140012, 19.159858, 21.177185, 23.182035, 25.199804, 27.212651,
        29.216700, 31.205516, 33.209544, 35.219265, 37.216348, 39.230041,
        41.242532, 43.245237, 45.252177, 47.220149, 49.236398, 51.248283},
		
       {1.211565,  1.311895,  1.460779,  1.251060,  1.277766,  1.312986,
        1.716564,  2.098457,  2.459967,  2.805381,  3.143568,  3.475037,
        3.798101,  4.120511,  4.438261,  4.753240,  5.068811,  5.378748,
        5.691811,  8.755126,  11.787875, 14.802370, 17.808191, 20.817430,
        23.841264, 26.825406, 29.846444, 32.848399, 35.844798, 38.838005,
        41.835642, 44.807681, 47.874351, 50.846945, 53.843309, 56.831403,
        59.850411, 62.868051, 65.831786, 68.818846, 71.820014, 74.862193},
		
       {1.214203,  1.316602,  1.475982,  1.273027,  1.306867,  1.349644,
        1.807364,  2.254934,  2.688736,  3.113848,  3.533751,  3.945386,
        4.358664,  4.765982,  5.172834,  5.580771,  5.985243,  6.394095,
        6.800004,  10.816307, 14.825071, 18.828409, 22.833925, 26.835960,
        30.840218, 34.822678, 38.821903, 42.845150, 46.865463, 50.844324,
        54.834713, 58.883133, 62.837446, 66.850908, 70.830658, 74.871135,
        78.816211, 82.872843, 86.830251, 90.827108, 94.868737, 98.851579},
		
       {1.223224,  1.305364,   1.282591,   1.302511,   1.343813,   1.391583,
        1.915889,  2.438440,   2.954782,   3.463715,   3.972178,   4.475866,
        4.981030,  5.484776,   5.988101,   6.489089,   6.987325,   7.492044,
        7.992565,  13.002149,  18.004261,  23.007252,  27.992595,  32.997420,
        38.007604, 42.995294,  47.960506,  53.006768,  57.998229,  62.978091,
        67.990712, 73.025313,  77.988608,  82.992045,  87.948481,  93.020257,
        98.037134, 103.005688, 107.968164, 112.955753, 118.091016, 123.064706},
		
       {1.237303,   1.297838,   1.299703,   1.336018,   1.386345,   1.440389,
        2.038549,   2.643681,   3.246379,   3.847373,   4.449428,   5.049205,
        5.648647,   6.246004,   6.847078,   7.447189,   8.042671,   8.641741,
        9.246661,   15.240034,  21.245192,  27.235472,  33.234248,  39.222670,
        45.239112,  51.237171,  57.234173,  63.224968,  69.264088,  75.216487,
        81.220146,  87.247497,  93.207614,  99.264302,  105.290686, 111.221248,
        117.130732, 123.207067, 129.211764, 135.248975, 141.199774, 147.221067},
		
       {1.255781,   1.301398,   1.326024,   1.373377,   1.431998,   1.494301,
        2.170349,   2.863678,   3.556875,   4.254405,   4.951657,   5.648337,
        6.344185,   7.042540,   7.738373,   8.438037,   9.135133,   9.842018,
        10.534453,  17.527653,  24.518568,  31.522017,  38.530695,  45.531330,
        52.535863,  59.486804,  66.539248,  73.511258,  80.457846,  87.477414,
        94.477479,  101.512152, 108.536001, 115.476406, 122.473642, 129.513069,
        136.476210, 143.513249, 150.550130, 157.499933, 164.525677, 171.621905},
		
       {1.276646,   1.315435,   1.357289,   1.415508,   1.481102,   1.551490,
        2.313390,   3.097829,   3.885637,   4.678081,   5.472808,   6.266119,
        7.063419,   7.858818,   8.656639,   9.454800,   10.253293,  11.045166,
        11.850929,  19.840428,  27.836272,  35.842002,  43.822372,  51.811789,
        59.830339,  67.850225,  75.868274,  83.871893,  91.824629,  99.842206,
        107.837178, 115.747088, 123.801923, 131.879473, 139.904745, 147.825008,
        155.800340, 163.858451, 171.872588, 179.842903, 187.749487, 195.829631},
		
       {1.296820,   1.336911,   1.392021,   1.460964,   1.534856,   1.613478,
        2.460424,   3.337350,   4.224161,   5.113733,   6.005077,   6.902279,
        7.794092,   8.695950,   9.585411,   10.488258,  11.382026,  12.283516,
        13.176733,  22.163355,  31.175048,  40.148876,  49.169087,  58.188614,
        67.159212,  76.140945,  85.190444,  94.138865,  103.180710, 112.151019,
        121.169030, 130.155468, 139.123738, 148.199908, 157.191572, 166.110967,
        175.158204, 184.278367, 193.189233, 202.156366, 211.208459, 220.106192},
		
       {1.314956,   1.362826,   1.430397,   1.507732,   1.590281,   1.677336,
        2.615813,   3.587581,   4.569082,   5.561364,   6.549072,   7.547573,
        8.537343,   9.537660,   10.527682,  11.533997,  12.527410,  13.528436,
        14.527811,  24.523978,  34.511355,  44.495985,  54.522808,  64.523901,
        74.496739,  84.506243,  94.476834,  104.521732, 114.526516, 124.492931,
        134.507726, 144.512764, 154.469854, 164.565066, 174.511718, 184.505784,
        194.563024, 204.550301, 214.551573, 224.414200, 234.453502, 244.490757},
		
       {1.377952,   1.458798,   1.554977,   1.660817,   1.772187,   1.886016,
        3.108236,   4.371942,   5.654082,   6.943083,   8.236469,   9.529075,
        10.822614,  12.119244,  13.411976,  14.715629,  16.011403,  17.304457,
        18.609621,  31.606019,  44.629317,  57.606470,  70.570352,  83.623095,
        96.649602,  109.608422, 122.636289, 135.528202, 148.553542, 161.627864,
        174.639075, 187.644904, 200.648839, 213.598369, 226.691110, 239.567775,
        252.597781, 265.726680, 278.632619, 291.690736, 304.684948, 317.539261},
		
       {1.484981,   1.607807,   1.744091,   1.888228,   2.038241,   2.191676,
        3.805323,   5.474649,   7.158217,   8.843498,   10.533764,  12.229865,
        13.922527,  15.619515,  17.316468,  19.033038,  20.717065,  22.416868,
        24.117551,  41.117558,  58.087372,  75.122262,  92.070802,  109.109606,
        126.060604, 143.111422, 160.052720, 176.993527, 193.966219, 210.996823,
        228.048774, 244.927949, 262.217855, 279.071730, 295.979342, 313.206434,
        329.907372, 347.043429, 364.039406, 381.250446, 398.166340, 414.966776},
		
       {1.576173,   1.731248,   1.898551,   2.074172,   2.252728,
        2.436570,   4.353815,   6.322445,   8.313083,   10.294745,
        12.287411,  14.277241,  16.277280,  18.268876,  20.271681,
        22.276521,  24.263926,  26.265196,  28.259014,  48.256424,
        68.222320,  88.220667,  108.266808, 128.276115, 148.278550,
        168.292247, 188.270702, 208.207174, 228.281385, 248.326064,
        268.274824, 288.273748, 308.305930, 328.090806, 348.214194,
        368.297940, 388.180241, 408.444446, 428.281276, 448.256689,
        468.212324, 488.220129}
		}};

// Read from skew_table.dat;
std::array<std::array<double, 42>, 14> skews = {
      {{9.120214, 11.475627, 38.412414, 2.221504, 1.684393, 1.446896, 0.781268,
        0.600871, 0.508428,  0.446559,  0.403048, 0.369343, 0.343748, 0.322870,
        0.304779, 0.289879,  0.276274,  0.265372, 0.255227, 0.191968, 0.160551,
        0.140669, 0.124834,  0.115103,  0.105681, 0.099399, 0.092551, 0.088156,
        0.082999, 0.080240,  0.076111,  0.073012, 0.071480, 0.068041, 0.065188,
        0.063611, 0.062579,  0.059057,  0.057525, 0.056310, 0.055544, 0.053913},
       {9.145098, 11.542642, 38.444059, 2.215191, 1.685763, 1.446827, 0.785224,
        0.600837, 0.504202,  0.441256,  0.399798, 0.366120, 0.340445, 0.319289,
        0.299703, 0.285555,  0.271096,  0.259854, 0.250297, 0.185038, 0.152457,
        0.130736, 0.115380,  0.103652,  0.095467, 0.087519, 0.081429, 0.075389,
        0.070460, 0.066910,  0.063430,  0.060366, 0.057555, 0.054277, 0.051492,
        0.050459, 0.048013,  0.045600,  0.044636, 0.042930, 0.040863, 0.040592},
       {9.130929, 11.496395, 38.553494, 2.213733, 1.686796, 1.444697, 0.773524,
        0.587195, 0.491120,  0.427417,  0.383929, 0.349077, 0.322633, 0.300579,
        0.282022, 0.265832,  0.250816,  0.239020, 0.228564, 0.160977, 0.124857,
        0.103811, 0.088318,  0.078253,  0.068526, 0.061949, 0.056645, 0.051888,
        0.047442, 0.044319,  0.042033,  0.038364, 0.036582, 0.034254, 0.032967,
        0.030693, 0.029447,  0.027500,  0.027629, 0.025326, 0.023565, 0.024128},
       {9.090881, 11.387960, 36.486987, 2.138348, 1.651012, 1.419899, 0.750984,
        0.567105, 0.468460,  0.404457,  0.358641, 0.324854, 0.296326, 0.273723,
        0.254571, 0.238595,  0.224091,  0.211870, 0.200157, 0.133642, 0.100628,
        0.080240, 0.067621,  0.057301,  0.050274, 0.045249, 0.040040, 0.036396,
        0.033308, 0.031502,  0.029914,  0.027198, 0.026498, 0.023856, 0.022186,
        0.022131, 0.021086,  0.019331,  0.018129, 0.017506, 0.017330, 0.015956},
       {8.495180, 10.051448, 30.876760, 1.985017, 1.581239, 1.371586, 0.722252,
        0.536916, 0.439019,  0.374528,  0.328893, 0.293841, 0.266643, 0.243633,
        0.224536, 0.209399,  0.195256,  0.183810, 0.174124, 0.110965, 0.081662,
        0.064387, 0.053649,  0.045021,  0.040079, 0.034299, 0.031512, 0.028912,
        0.025870, 0.023838,  0.022831,  0.021006, 0.020104, 0.018436, 0.016838,
        0.016731, 0.016235,  0.015160,  0.013796, 0.013969, 0.012440, 0.012756},
       {7.280436, 6.954720, 2.759347, 1.828208, 1.501224, 1.304616, 0.684929,
        0.504954, 0.406041, 0.342880, 0.299160, 0.264531, 0.237149, 0.217317,
        0.199300, 0.183890, 0.171015, 0.160512, 0.150267, 0.093537, 0.067263,
        0.054102, 0.044438, 0.037667, 0.032276, 0.028385, 0.025535, 0.023194,
        0.021282, 0.019795, 0.019279, 0.017462, 0.015565, 0.015138, 0.013917,
        0.013269, 0.013570, 0.012332, 0.011480, 0.010412, 0.009803, 0.009744},
       {6.044311, 4.464386, 2.270965, 1.683147, 1.408570, 1.231329, 0.647790,
        0.469020, 0.374580, 0.313057, 0.270302, 0.238628, 0.213000, 0.192970,
        0.176484, 0.162463, 0.150547, 0.139979, 0.130501, 0.079703, 0.057365,
        0.045737, 0.036707, 0.031797, 0.027383, 0.024354, 0.022312, 0.020679,
        0.017407, 0.016420, 0.014826, 0.014685, 0.013240, 0.012463, 0.012894,
        0.010440, 0.010327, 0.009738, 0.009333, 0.009624, 0.008553, 0.009300},
       {5.035575, 3.204568, 1.987774, 1.555399, 1.318778, 1.160449, 0.606904,
        0.436103, 0.343434, 0.287311, 0.246226, 0.215142, 0.192055, 0.171639,
        0.157195, 0.144339, 0.133287, 0.123859, 0.115028, 0.069635, 0.051185,
        0.039354, 0.031507, 0.026961, 0.023131, 0.020698, 0.019201, 0.017478,
        0.016249, 0.014521, 0.013224, 0.012654, 0.011854, 0.011207, 0.010067,
        0.010211, 0.008342, 0.008700, 0.008625, 0.007851, 0.006954, 0.007121},
       {4.237669, 2.575665, 1.795866, 1.443392, 1.233284, 1.092845, 0.570030,
        0.404245, 0.316789, 0.262220, 0.223466, 0.194478, 0.172904, 0.155156,
        0.141943, 0.130392, 0.120730, 0.111397, 0.102695, 0.063214, 0.045266,
        0.033946, 0.028248, 0.024592, 0.020644, 0.018409, 0.016099, 0.014638,
        0.013580, 0.012534, 0.011060, 0.010175, 0.009928, 0.009802, 0.009131,
        0.008032, 0.008216, 0.006900, 0.007915, 0.007030, 0.006235, 0.006205},
       {3.567612, 2.217888, 1.638238, 1.343229, 1.157161, 1.025149, 0.533331,
        0.375971, 0.293297, 0.240604, 0.204908, 0.178419, 0.157993, 0.141526,
        0.128039, 0.117845, 0.107844, 0.100090, 0.093206, 0.055927, 0.039632,
        0.031419, 0.025441, 0.022017, 0.017970, 0.017046, 0.013781, 0.013837,
        0.011788, 0.010844, 0.010719, 0.009848, 0.008781, 0.008666, 0.008390,
        0.007383, 0.006704, 0.006690, 0.006704, 0.005597, 0.005315, 0.006187},
       {3.025613, 1.971907, 1.508690, 1.253347, 1.085729, 0.964902, 0.500720,
        0.350103, 0.270796, 0.222378, 0.189370, 0.163883, 0.144245, 0.129899,
        0.117215, 0.106412, 0.098729, 0.091369, 0.084997, 0.050933, 0.035450,
        0.028079, 0.022752, 0.019002, 0.016494, 0.014958, 0.013209, 0.011917,
        0.010909, 0.011051, 0.009086, 0.008943, 0.008319, 0.007697, 0.007308,
        0.006254, 0.006289, 0.006687, 0.005169, 0.005813, 0.005401, 0.005271},
       {2.103528, 1.521551, 1.224627, 1.037122, 0.908020, 0.811763, 0.418322,
        0.287979, 0.220201, 0.178456, 0.150544, 0.130941, 0.114658, 0.101290,
        0.093054, 0.083796, 0.078241, 0.071278, 0.066739, 0.039318, 0.027836,
        0.021947, 0.017540, 0.015152, 0.012936, 0.011566, 0.010478, 0.008942,
        0.008706, 0.008533, 0.007591, 0.006950, 0.005415, 0.006408, 0.005202,
        0.005010, 0.005253, 0.005104, 0.004322, 0.005458, 0.004121, 0.004314},
       {1.536751, 1.182345, 0.975411, 0.837641, 0.736944, 0.660374, 0.337183,
        0.229715, 0.174118, 0.140625, 0.118463, 0.102568, 0.089017, 0.079632,
        0.072125, 0.066294, 0.059757, 0.055515, 0.051649, 0.030344, 0.020841,
        0.016917, 0.013317, 0.011891, 0.010315, 0.008493, 0.007762, 0.006624,
        0.007287, 0.006208, 0.005151, 0.005726, 0.004994, 0.004219, 0.004152,
        0.003915, 0.003437, 0.003389, 0.003397, 0.002747, 0.002961, 0.003400},
       {1.286577, 1.011802, 0.843325, 0.727349, 0.642881, 0.577038, 0.293274,
        0.199210, 0.150758, 0.121809, 0.101102, 0.087423, 0.077013, 0.067438,
        0.060997, 0.055508, 0.050853, 0.048553, 0.044361, 0.026014, 0.018513,
        0.014217, 0.011612, 0.009604, 0.007862, 0.007154, 0.005997, 0.006527,
        0.005605, 0.004194, 0.004040, 0.004002, 0.004299, 0.004607, 0.004030,
        0.003770, 0.002902, 0.003709, 0.002591, 0.001857, 0.002333, 0.002402}}};

template<class T>
class mor_des_t {
    // Read from l_sat_grid.dat
    
    //typename T::Interp2D const sig_interp;
    //typename T::Interp2D const skews_interp;
	
  public:
    mor_des_t() = default;
	
	mor_des_t(typename T::Interp2D* sig_int, typename T::Interp2D* skews_int):sig_interp(sig_int), skews_interp(skews_int) {}
	
    mor_des_t(double A,
              double B,
              double C,
              double sigma_i,
              double epsilon,
              double z_pivot)
      : _A(A)
      , _B(B)
      , _C(C)
      , _sigma_intr(sigma_i)
      , _epsilon(epsilon)
      , _z_pivot(z_pivot){}
	  
	~mor_des_t(){
		//just added
		//cudaFree(skews_interp);
		//cudaFree(sig_interp);
		//delete skews_interp;
		//delete sig_interp;
	}
	  
	__device__ __host__
    double
    operator()(double lt, double lnM, double zt) const
    {
	  //#ifdef  __CUDA_ARCH__
	  //printf("_cols:%lu _rows:%lu\n", sig_interp->_cols, sig_interp->_rows);
	  //printf("[%i](%i)computing mor_des %f, %f, %f\n", blockIdx.x, threadIdx.x, lt, lnM, zt);
	  //printf("[%i](%i)_z_pivot:%f _epsilon:%f\n", blockIdx.x, threadIdx.x, _z_pivot, _epsilon);
	  //#endif
	  
	  double term1 = (exp(lnM) - _A) / (_B - _A);
	   //#ifdef  __CUDA_ARCH__
	  //printf("[%i](%i)1st:%f\n",  blockIdx.x, threadIdx.x, pow(term1, _C));
	  //#endif
	  double pow_term1 = pow(term1, _C);
	  //#ifdef  __CUDA_ARCH__
	  //printf("[%i](%i)2nd:%f\n", blockIdx.x, threadIdx.x , pow((1.0 + zt) / (1.0 + _z_pivot), _epsilon));
	  //#endif
	  double  term1a = pow(term1, _C);
	  //double  term2a = pow((1.0 + zt) / (1.0 + _z_pivot), _epsilon);
	   double  term2a = 0.;
	  if(_epsilon<0){
		  term2a = 1./pow((1.0 + zt) / (1.0 + _z_pivot), -1.*_epsilon);
	  }
	  else{
		  term2a = pow((1.0 + zt) / (1.0 + _z_pivot), _epsilon);
	  }
	  
      //double const ltm = pow(term1, _C) *
       //                  pow((1.0 + zt) / (1.0 + _z_pivot), _epsilon);
	  double  ltm = term1a*term2a;
	 
	  //+0.04 gives different result, +.03 not
	  //#ifdef  __CUDA_ARCH__
	  //printf("[%i](%i) interpolating at %f, %f\n", blockIdx.x, threadIdx.x, _sigma_intr, ltm);
	  //#endif
	  
      double  _sigma = sig_interp->clamp(_sigma_intr, ltm);
	 
      
	  double  _skw = skews_interp->clamp(_sigma_intr, ltm);
      double  x = lt - ltm;
      double  erfarg = -1.0 * _skw * (x) / (sqrt(2.) * _sigma);
      double  erfterm = erfc(erfarg);
      return quad::gaussian(x, 0.0, _sigma) * erfterm;
    }
	
    friend std::ostream&
    operator<<(std::ostream& os, mor_des_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat;
      os << m._A << ' ' << m._B << ' ' << m._C << ' ' << m._sigma_intr << ' '
         << m._epsilon << ' ' << m._z_pivot;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, mor_des_t& m)
    {
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> vals_read = cosmosis::str_to_doubles(buffer);
      if (vals_read.size() == 6)
      {
		typename T::Interp2D *sig_interp /*m.sig_interp*/    = new typename T::Interp2D(test_sigintr, test_lsat, sig_skewnorml_flat);
		typename T::Interp2D *skews_interp/*m.skews_interp*/ = new typename T::Interp2D(test_sigintr, test_lsat, skews);
      
	    m = mor_des_t(sig_interp, skews_interp);
		m._A = vals_read[0];
        m._B = vals_read[1];
        m._C = vals_read[2];
        m._sigma_intr = vals_read[3];
        m._epsilon = vals_read[4];
        m._z_pivot = vals_read[5];	
	  }
      else
      {
        is.setstate(std::ios_base::failbit);
      }
      return is;
    }
	
	//private:
	//std::shared_ptr<typename T::Interp2D const> sig_interp;
	//std::shared_ptr<typename T::Interp2D const> skews_interp;
	
	typename T::Interp2D*  sig_interp;
    typename T::Interp2D* skews_interp;
	
    double _A = 0.0;
    double _B = 0.0;
    double _C = 0.0;
    double _sigma_intr = 0.0;
    double _epsilon = 0.0;
    double _z_pivot = 0.0;

  };

/*TEST_CASE("MOR_DES_t utilizing gaussian on gpu and std::array initialization")
{
	double const lt = 0x1.b8p+4;
	double const lnM = 0x1.0cp+5;
	double const zt = 0x1.cccccccccccccp-2;
	y3_cluster::MOR_DES_t mor = make_from_file<y3_cluster::MOR_DES_t>("data/MOR_DES_t.dump");
	mor_des_t<GPU> dmor       = make_from_file<mor_des_t<GPU>>("data/MOR_DES_t.dump");
	
	double result = mor(lt, lnM, zt);
	double cpu_result = dmor(lt, lnM, zt);
	CHECK(result == cpu_result);
	
	mor_des_t<GPU> *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(mor_des_t<GPU>));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &dmor, sizeof(mor_des_t<GPU>));
		
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
		
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, lt, lnM, zt, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
	printf("true:%.25f\n", result);
	printf("cpu :%.25f\n", cpu_result);
	printf("gpu :%.25f\n\n", *gpu_result);
	
	printf("true:%e\n", result);
	printf("cpu :%e\n", cpu_result);
	printf("gpu :%e\n\n", *gpu_result);
	
	printf("true:%a\n", result);
	printf("cpu :%a\n", cpu_result);
	printf("gpu :%a\n", *gpu_result);
	printf("------------------------------\n");
}*/

  class roffset_t {
  public:
    roffset_t() = default;

    explicit roffset_t(double tau) : _tau(tau) {}
	
	__device__ __host__
    double
    operator()(double x) const
    {
      // eq. 36
      return x / _tau / _tau * exp(-x / _tau);
    }

    friend std::ostream&
    operator<<(std::ostream& os, roffset_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << m._tau;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, roffset_t& m)
    {
      std::string buffer;
      std::getline(is, buffer);
      if (!is) return is;
      m._tau = std::stod(buffer);
      return is;
    }

  private:
    double _tau = 0.0;
  };
  
/*TEST_CASE("ROFFSET_t")
{
	double const rmis = 0x1p+0;
	y3_cluster::ROFFSET_t roffset 	= make_from_file<y3_cluster::ROFFSET_t>("data/ROFFSET_t.dump");
	double result = roffset(rmis);
	roffset_t droffset = make_from_file<roffset_t>("data/ROFFSET_t.dump");
	double cpu_result = droffset(rmis);
	CHECK(result == cpu_result);
	
	roffset_t *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(roffset_t));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &droffset, sizeof(roffset_t));
		
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
		
	testKernel<roffset_t><<<1,1>>>(dhmf2, rmis, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
	
	cudaFree(dhmf2);
	cudaFree(gpu_result);
}*/

 class ez_sq {
  public:
    ez_sq() = default;
	
	__host__ __device__
    ez_sq(double omega_m, double omega_l, double omega_k)
      : _omega_m(omega_m), _omega_l(omega_l), _omega_k(omega_k)
    {}
	
	__host__ __device__
    double
    operator()(double z) const
    {
      // NOTE: this is valid only for \Lambda CDM cosmology, not wCDM
      double const zplus1 = 1.0 + z;
      return (_omega_m * zplus1 * zplus1 * zplus1 + _omega_k * zplus1 * zplus1 +
              _omega_l);
    }

	friend std::ostream&
    operator<<(std::ostream& os, ez_sq const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << m._omega_m << ' ' << m._omega_l << ' '
         << m._omega_k;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, ez_sq& m)
    {
      assert(is.good());
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> const vals_read = cosmosis::str_to_doubles(buffer);
      if (vals_read.size() == 3)
      {
        m._omega_m = vals_read[0];
        m._omega_l = vals_read[1];
        m._omega_k = vals_read[2];
      }
      else
      {
        is.setstate(std::ios_base::failbit);
      };
      return is;
    }

  private:
    double _omega_m = 0.0;
    double _omega_l = 0.0;
    double _omega_k = 0.0;
  };

class ez {
  public:
    ez() = default;
	
	__host__ __device__
    ez(double omega_m, double omega_l, double omega_k)
      : _ezsq(omega_m, omega_l, omega_k)
    {}

	__host__ __device__
    double
    operator()(double z) const
    {
      auto const sqr = _ezsq(z);
      return sqrt(sqr);
    }
	
    friend std::ostream&
    operator<<(std::ostream& os, ez const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat;
      os << m._ezsq;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, ez& m)
    {
      assert(is.good());
      is >> m._ezsq;
      return is;
    }

  private:
    ez_sq _ezsq;
  };

template<class T>
class dv_do_dz_t {
  public:
    dv_do_dz_t() = default;
	
	__host__ __device__
    dv_do_dz_t(typename T::Interp1D* da, ez ezt, double h)
      : _da(da), _ezt(ezt), _h(h)
    {}

    using doubles = std::vector<double>;

	__host__ __device__
    double
    operator()(double zt) const
    {
      double const da_z = _da->eval(zt); // da_z needs to be in Mpc
      // Units: (Mpc/h)^3
      // 2997.92 is Hubble distance, c/H_0
      return 2997.92 * (1.0 + zt) * (1.0 + zt) * da_z * _h * da_z * _h /
             _ezt(zt);
    }
	
	/*
    friend std::ostream&
    operator<<(std::ostream& os, dv_do_dz_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << *m._da << '\n' << m._ezt << '\n' << m._h;
      os.flags(old_flags);
      return os;
    }*/

    friend std::istream&
    operator>>(std::istream& is, dv_do_dz_t& m)
    {
      assert(is.good());
      //auto da = std::make_shared<typename T::Interp1D>();
	  typename T::Interp1D *da = new typename T::Interp1D;
      is >> *da;
      if (!is) return is;
      ez _ez;
      is >> _ez;
      if (!is) return is;
      std::string buffer;
      std::getline(is, buffer);
      if (!is) return is;
      double const h = std::stod(buffer);
      m = dv_do_dz_t(da, _ez, h);
      return is;
    }

  private:
    //std::shared_ptr<typename T::Interp1D const> _da;
	typename T::Interp1D* _da; 
    ez _ezt;
    double _h;
  };
  
/*TEST_CASE("dv_do_dz_t")
 {
	 double const zt = 0x1.cccccccccccccp-2;
	 y3_cluster::DV_DO_DZ_t dv_do_dz 	= make_from_file<y3_cluster::DV_DO_DZ_t>("data/DV_DO_DZ_t.dump");
	 double result = dv_do_dz(zt);
	 
	 dv_do_dz_t<GPU> d_dv_do_dz = make_from_file<dv_do_dz_t<GPU>>("data/DV_DO_DZ_t.dump");
	 double cpu_result = d_dv_do_dz(zt);
	 CHECK(result == cpu_result);

	 dv_do_dz_t<GPU> *dhmf2;
	 cudaMallocManaged((void**)&dhmf2, sizeof(dv_do_dz_t<GPU>));
	 cudaDeviceSynchronize();
	 memcpy(dhmf2, &d_dv_do_dz, sizeof(dv_do_dz_t<GPU>));
	 CHECK(dhmf2->operator()(zt) == result);
	 
	 double* gpu_result;
	 cudaMallocManaged((void**)&gpu_result, sizeof(double));
	 
	 testKernel<dv_do_dz_t<GPU>><<<1,1>>>(dhmf2, zt, gpu_result);
	 cudaDeviceSynchronize();
	 CHECK(*gpu_result == result);
	 
	 cudaFree(dhmf2);
	 cudaFree(gpu_result);
 }*/
 
 template<class T>
 class sig_sum {
  private:

    //std::shared_ptr<typename T::Interp2D const> _sigma1;
   // std::shared_ptr<typename T::Interp2D const> _sigma2;
    //std::shared_ptr<typename T::Interp2D const> _bias;
	typename T::Interp2D* _sigma1;
	typename T::Interp2D* _sigma2;
	typename T::Interp2D* _bias;
  public:
    using doubles = std::vector<double>;
    sig_sum() = default;
	
    sig_sum(typename T::Interp2D* sigma1,
            typename T::Interp2D* sigma2,
            typename T::Interp2D* bias)
      : _sigma1(sigma1), _sigma2(sigma2), _bias(bias)
    {}
	
	~sig_sum(){
		//just added
		//cudaFree(_sigma1);
		//cudaFree(_sigma2);
		//cudaFree(_bias);
		//delete _sigma1;
		//delete _sigma2;
		//delete _bias;
	}
	
	__host__ __device__
    double
    operator()(double r, double lnM, double zt) const
    /*r in h^-1 Mpc */ /* M in h^-1 M_solar, represents M_{200} */
    {
      double _sig_1 = _sigma1->clamp(r, lnM);
      double _sig_2 = _bias->clamp(zt, lnM) * _sigma2->clamp(r, zt);
      // TODO: h factor?
      // if (del_sig_1 >= del_sig_2) {
      // return (1.+zt)*(1.+zt)*(1.+zt)*(_sig_1+_sig_2);
      return (_sig_1 + _sig_2);
      /*} else {
        return 1e12*(1.+zt)*(1.+zt)*(1.+zt)*del_sig_2;
      } */
    }

    friend std::ostream&
    operator<<(std::ostream& os, sig_sum const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << *m._sigma1 << '\n' << *m._sigma2 << '\n' << *m._bias;
      os.flags(old_flags);
      return os;
    }
	
    friend std::istream&
    operator>>(std::istream& is, sig_sum& m)
    {
      //auto sigma1 = std::make_shared<typename T::Interp2D>();
	  typename T::Interp2D *sigma1 = new typename T::Interp2D;
      is >> *sigma1;
      if (!is) return is;
      //auto sigma2 = std::make_shared<typename T::Interp2D>();
	  typename T::Interp2D *sigma2 = new typename T::Interp2D;
      is >> *sigma2;
      if (!is) return is;
      //auto bias = std::make_shared<typename T::Interp2D>();
	  typename T::Interp2D *bias = new typename T::Interp2D;
      is >> *bias;
      if (!is) return is;
      m = sig_sum(sigma1, sigma2, bias);
      return is;
    }
  };
  
/*TEST_CASE("SIG_SUM")
{
	double const theta = 0x1.921fb54442eeap+1;
	double const radius_ = 0x1p+0;
	double const rmis = 0x1p+0;
	double const lnM = 0x1.0cp+5;
	double const zt = 0x1.cccccccccccccp-2;
	double const scaled_Rmis = std::sqrt(radius_ * radius_ + rmis * rmis +
                                       2 * rmis * radius_ * std::cos(theta));
	y3_cluster::SIG_SUM sigma = make_from_file<y3_cluster::SIG_SUM>("data/SIG_SUM.dump");
	double result  = sigma(scaled_Rmis, lnM, zt);
	sig_sum<GPU> d_sigma = make_from_file<sig_sum<GPU>>("data/SIG_SUM.dump");
	double cpu_result = d_sigma(scaled_Rmis, lnM, zt);
	CHECK(result == cpu_result);
	
	sig_sum<GPU> *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(sig_sum<GPU>));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &d_sigma, sizeof(sig_sum<GPU>));
	CHECK(dhmf2->operator()(scaled_Rmis, lnM, zt) == result);
	 
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
	 
	testKernel<sig_sum<GPU>><<<1,1>>>(dhmf2, scaled_Rmis, lnM, zt, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
	 
	cudaFree(dhmf2);
	cudaFree(gpu_result);
}*/

class  lo_lc_t{
  public:
    lo_lc_t() = default;

    lo_lc_t(double alpha, double a, double b, double R_lambda)
      : _alpha(alpha), _a(a), _b(b), _R_lambda(R_lambda)
    {}

	__host__ __device__
    double
    operator()(double lo, double lc, double R_mis) const
    {
      /* eq. (35) */
      double x = R_mis / _R_lambda;
      double y = lo / lc;
      double mu_y = exp(-x * x / _alpha / _alpha);
      double sigma_y = _a * atan(_b * x);
      // Need 1/lc scaling for total probability = 1
      return quad::gaussian(y, mu_y, sigma_y) / lc;
    }

    friend std::ostream&
    operator<<(std::ostream& os, lo_lc_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << m._alpha << ' ' << m._a << ' ' << m._b << ' '
         << m._R_lambda;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, lo_lc_t& m)
    {
      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> const vals_read = cosmosis::str_to_doubles(buffer);
      if (vals_read.size() == 4)
      {
        m._alpha = vals_read[0];
        m._a = vals_read[1];
        m._b = vals_read[2];
        m._R_lambda = vals_read[3];
      }
      else {
        is.setstate(std::ios_base::failbit);
      }
      return is;
    }

  private:
    double _alpha = 0.0;
    double _a = 0.0;
    double _b = 0.0;
    double _R_lambda = 0.0;
};

/*TEST_CASE("LO_LC_t")
{
	double const lo = 0x1.9p+4;
    double const lc = 0x1.b8p+4;
	double const rmis = 0x1p+0;
	y3_cluster::LO_LC_t lo_lc = make_from_file<y3_cluster::LO_LC_t>("data/LO_LC_t.dump");
	double result = lo_lc(lo, lc, rmis);
	lo_lc_t d_lo_lc = make_from_file<lo_lc_t>("data/LO_LC_t.dump");
	double cpu_result = d_lo_lc(lo, lc, rmis);
	CHECK(result == cpu_result);
	
	lo_lc_t *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(lo_lc_t));
	cudaDeviceSynchronize();
	memcpy(dhmf2, &d_lo_lc, sizeof(lo_lc_t));
	CHECK(dhmf2->operator()(lo, lc, rmis) == result);
	 
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
	 
	testKernel<lo_lc_t><<<1,1>>>(dhmf2, lo, lc, rmis, gpu_result);
	cudaDeviceSynchronize();
	CHECK(*gpu_result == result);
	 
	cudaFree(dhmf2);
	cudaFree(gpu_result);
}*/

template<class T>
class integral {
public:
  using grid_point_t = std::array<double, 3>; // we only vary radius.

//private:
  using volume_t = cubacpp::IntegrationVolume<7>;

  // State obtained from configuration. These things should be set in the
  // constructor.
  // <none in this example>

  // State obtained from each sample.
  // If there were a type X that did not have a default constructor,
  // we would use optional<X> as our data member.
  
   int_lc_lt_des_t<T> 	lc_lt;
   mor_des_t<T>			mor;
   omega_z_des<T>		omega_z;
  dv_do_dz_t<T> 		dv_do_dz;
  hmf_t<T> 				hmf;
  int_zo_zt_des_t 		int_zo_zt;
  roffset_t 			roffset;
  lo_lc_t 				lo_lc;
  sig_sum<T> 			sigma;

  // State set for current 'bin' to be integrated.
  double zo_low_ = 0.0;
  double zo_high_ = 0.0;
  double radius_ = 0.0;

public:
	// Default c'tor just for testing outside of CosmoSIS.
	integral()
	{
		mor 	 = make_from_file<mor_des_t<GPU>>("data/MOR_DES_t.dump");
		dv_do_dz = make_from_file<dv_do_dz_t<GPU>>("data/DV_DO_DZ_t.dump");
		hmf 	 = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
		roffset  = make_from_file<roffset_t>("data/ROFFSET_t.dump");
		sigma  	 = make_from_file<sig_sum<GPU>>("data/SIG_SUM.dump");
		lo_lc 	 = make_from_file<lo_lc_t>("data/LO_LC_t.dump");
	}

	// Set any data members from values read from the current sample.
	// Do not attempt to copy the sample!.
	void set_sample(int_lc_lt_des_t<T> const& int_lc_lt_in,
                  mor_des_t<T> const& mor_in,
                  omega_z_des<T> const& omega_z_in,
                  dv_do_dz_t<T> const& dv_do_dz_in,
                  hmf_t<T> const& hmf_in,
                  int_zo_zt_des_t const& int_zo_zt_in,
                  roffset_t const& roffset_in,
                  lo_lc_t const& lo_lc_in,
                  sig_sum<T> const& sig_sum_in)
	{
		
		mor = std::move(mor_in);
		dv_do_dz = std::move(dv_do_dz_in);
		lc_lt = std::move(int_lc_lt_in);
		omega_z = std::move(omega_z_in);
		
		hmf = std::move(hmf_in);
		int_zo_zt = std::move(int_zo_zt_in);
		roffset = std::move(roffset_in);
		lo_lc = std::move(lo_lc_in);
		sigma = std::move(sig_sum_in);
	}

  // Set the data for the current bin.
	void set_grid_point(grid_point_t const& grid_point)
	{
		radius_ = grid_point[2];
		zo_low_ = grid_point[0];
		zo_high_ = grid_point[1];
	}

	// The function to be integrated. All arguments to this function must be of
	// type double, and there must be at least two of them (because our
	// integration routine does not work for functions of one variable). The
	// function is const because calling it does not change the state of the
	// object.
	__device__
	double operator()(double lo,
                    double lc,
                    double lt,
                    double zt,
                    double lnM,
                    double rmis,
                    double theta) const{
		
		double const mor_des = (mor)(lt, lnM, zt);
		//printf("mor_des:%f\n", mor_des);
		double const common_term = (roffset)(rmis) * (lo_lc)(lo, lc, rmis) *
                             (lc_lt)(lc, lt, zt) * mor_des *
                             (dv_do_dz)(zt) * (hmf)(lnM, zt) *
                             (omega_z)(zt) / 2.0 / 3.1415926535897;
		//printf("common_term:%f\n", common_term);					 
		double const scaled_Rmis = sqrt(radius_ * radius_ + rmis * rmis +
                                       2 * rmis * radius_ * cos(theta));
		//printf("scaled_Rmis:%f\n", scaled_Rmis);
		double const val = (sigma)(scaled_Rmis, lnM, zt) *
                   (int_zo_zt)(zo_low_, zo_high_, zt) * common_term;
		//if(blockIdx.x == 0)
		//	printf("[%i](%i) %f, %f, %f, %f, %f, %f, %f = %.20f\n", blockIdx.x, threadIdx.x, lo, lc, lt, zt, lnM, rmis, theta, val);
		/*if(threadIdx.x == 0)
			printf("(%i) %f, %f, %f, %f, %f, %f, %f = %.20f\n", 
																 threadIdx.x,
																 lo,
																 lc,
																 lt, 
																 zt,
																 lnM,
																 rmis,
																 theta, 
																 val);*/
								
		//printf("[%i](%i) val:%a\n", val);
		//printf("    roffset:%a\n", (roffset)(rmis));
		//printf("    lo_lc:%a\n", (lo_lc)(lo, lc, rmis));
		//printf("    lc_lt:%a\n", (lc_lt)(lc, lt, zt));
		//printf("    mor:%val\n", val);
		//printf("    dv_do_dz:%a\n", (dv_do_dz)(zt));
		//printf("    omega_z:%a\n", (omega_z)(zt));
		//printf("    hmf:%a\n", (hmf)(lnM, zt));
		
		//printf("scaled_Rmis:%a\n", scaled_Rmis);
		//printf("Sigma:%a\n", (sigma)(scaled_Rmis, lnM, zt));
		//printf("int_zo_zt:%a\n", (int_zo_zt)(zo_low_, zo_high_, zt));
		//printf("[%i](%i) %f, %f, %f, %f, %f, %f, %f, %.20f\n", blockIdx.x, threadIdx.x, lo, lc, lt, zt, lnM, rmis, theta, val);
		return val;		
	}
};

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

template <typename ALG, typename F>
bool
time_and_call_alt(ALG const& a, F f, double epsrel, double correct_answer, std::string algname, int _final=0)
{
  using MilliSeconds = std::chrono::duration<double, std::chrono::milliseconds::period>;
  // We make epsabs so small that epsrel is always the stopping condition.
  double constexpr epsabs = 1.0e-40;
  cubacpp::array<7> lows  = {20., 5.,  5., .15,  29., 0., 0.};
  cubacpp::array<7> highs = {30., 50., 50.,.75,  38., 1., 6.28318530718};
  cubacpp::integration_volume_for_t<F> vol(lows, highs);
  
  auto t0 = std::chrono::high_resolution_clock::now();
  printf("time-and-call\n");
  auto res = a.integrate(f, epsrel, epsabs, vol);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double absolute_error = std::abs(res.value - correct_answer);
  bool const good = (res.status == 0);
  int converge = !good;
  std::cout.precision(15); 
  std::cout<<algname<<","
		   <<std::to_string(correct_answer)<<","
			<<epsrel<<","
			<<epsabs<<","
			<<std::to_string(res.value)<<","
			<<std::to_string(res.error)<<","
			<<res.nregions<<","
			<<res.status<<","
			<<_final<<","
			<<dt.count()<<std::endl;
  if(res.status == 0)
	return true;
  else
	return false;
}

template <typename F>
bool
time_and_call(std::string id,
              F integrand,
              double epsrel,
              double true_value,
              char const* algname,
              std::ostream& outfile,
              int _final = 0)
{
	//printf("time_and_call d_integrand Mor des cols:%lu\n", integrand.mor.sig_interp->_cols);
	//printf("inside time and call\n");
	//printf("time_and_call d_integrand Mor des cols:%lu\n", integrand.mor.sig_interp->_cols);
  using MilliSeconds =
    std::chrono::duration<double, std::chrono::milliseconds::period>;
  double constexpr epsabs = 1.0e-40;

  double lows[] =  {20., 5.,  5., .15,  29., 0., 0.};
  double highs[] = {30., 50., 50.,.75,  38., 1., 6.28318530718};

  constexpr int ndim = 7;
  quad::Volume<double, ndim> vol(lows, highs);
  int const key = 0;
  int const verbose = 0;
  int const numdevices = 1;
  quad::Cuhre<double, ndim> alg(0, nullptr, key, verbose, numdevices);
	
  int outfileVerbosity = 0;
  constexpr int phase_I_type = 0; // alternative phase 1

  auto const t0 = std::chrono::high_resolution_clock::now();
  
  cuhreResult const result = alg.integrate<F>(integrand, epsrel, epsabs, &vol, outfileVerbosity, _final, phase_I_type);
  
  MilliSeconds dt = std::chrono::high_resolution_clock::now() - t0;
  double const absolute_error = std::abs(result.estimate - true_value);
  bool good = false;

  if (result.status == 0 || result.status == 2) {
    good = true;
  }
  outfile.precision(15);
  outfile << std::fixed << id << ",\t" << std::scientific << true_value << ",\t"
          << std::scientific << epsrel << ",\t\t\t" << std::scientific
          << epsabs << ",\t" << std::scientific << result.estimate << ",\t"
          << std::scientific << result.errorest << ",\t" << std::fixed
          << result.nregions << ",\t" << std::fixed << result.status << ",\t"
          << _final << ",\t" << dt.count() << std::endl;

  return good;
}

template <typename F>
void innerWrapper(F d_integrand){
	printf("innerWrapper d_integrand Mor des cols:%lu\n", d_integrand.mor.sig_interp->_cols);
	integral<GPU> *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(integral<GPU>));
	memcpy(dhmf2, &d_integrand, sizeof(integral<GPU>));
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
	
	double const lo = 0x1.9p+4;
    double const lc = 0x1.b8p+4;
    double const lt = 0x1.b8p+4;
    double const zt = 0x1.cccccccccccccp-2;
    double const lnM = 0x1.0cp+5;
    double const rmis = 0x1p+0;
    double const theta = 0x1.921fb54442eeap+1;
	
    double const radius_ = 0x1p+0;
    double const zo_low_ = 0x1.999999999999ap-3;
    double const zo_high_ = 0x1.6666666666666p-2;
	testKernel<integral<GPU>><<<1,1>>>(dhmf2, lo, lc, lt, zt, lnM, rmis, theta, gpu_result);
	cudaDeviceSynchronize();
	printf("End of kernel wrapper\n");
}

template <typename F>
void kernel_wrapper(F d_integrand){
	printf("kernel_wrapper d_integrand Mor des cols:%lu\n", d_integrand.mor.sig_interp->_cols);
	//innerWrapper(d_integrand);
}

class test {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m, double p)
  {
    return sin(x + y + z + k + l + m);
  }
};

int
main()
{

    //TEST_CASE("integral call"){
	//SigmaMiscentY1ScalarIntegrand integrand2;
	//test Testobj;
	
	//time_and_call_vegas(Testobj);

	printf("Final Test Case\n");
	double const lo = 0x1.9p+4;
    double const lc = 0x1.b8p+4;
    double const lt = 0x1.b8p+4;
    double const zt = 0x1.cccccccccccccp-2;
    double const lnM = 0x1.0cp+5;
    double const rmis = 0x1p+0;
    double const theta = 0x1.921fb54442eeap+1;
	
    double const radius_ = 0x1p+0;
    double const zo_low_ = 0x1.999999999999ap-3;
    double const zo_high_ = 0x1.6666666666666p-2;
	
	y3_cluster::INT_LC_LT_DES_t lc_lt;     // we want the default
	y3_cluster::OMEGA_Z_DES	 	omega_z;       // we want the default
	y3_cluster::INT_ZO_ZT_DES_t int_zo_zt; // we want the default
	
	y3_cluster::MOR_DES_t 	mor 		= make_from_file<y3_cluster::MOR_DES_t>("data/MOR_DES_t.dump");
	y3_cluster::DV_DO_DZ_t 	dv_do_dz 	= make_from_file<y3_cluster::DV_DO_DZ_t>("data/DV_DO_DZ_t.dump");
	y3_cluster::HMF_t 		hmf 		= make_from_file<y3_cluster::HMF_t>("data/HMF_t.dump");
	y3_cluster::ROFFSET_t 	roffset 	= make_from_file<y3_cluster::ROFFSET_t>("data/ROFFSET_t.dump");
	y3_cluster::SIG_SUM 	sig_sum 	= make_from_file<y3_cluster::SIG_SUM>("data/SIG_SUM.dump");
	y3_cluster::LO_LC_t 	lo_lc 		= make_from_file<y3_cluster::LO_LC_t>("data/LO_LC_t.dump");
	
	SigmaMiscentY1ScalarIntegrand integrand;
	integrand.set_sample(lc_lt, mor, omega_z, dv_do_dz, hmf, int_zo_zt, roffset, lo_lc, sig_sum);
	integrand.set_grid_point({zo_low_, zo_high_, radius_});
	double result = integrand(lo, lc, lt, zt, lnM, rmis, theta);
	time_and_call_vegas(integrand);
	return 0;
	//time_and_call_vegas(integrand);
    cubacores(0, 0);

	unsigned long long constexpr mmaxeval = std::numeric_limits<unsigned long long>::max();
	std::cout<<"mmaxeval:"<<mmaxeval<<"\n";
										    
	unsigned long long constexpr maxeval = 1000 * 1000 * 1000;
	double const epsrel_min = 1.0e-12;
	cubacpp::Cuhre cuhre;
	int verbose = 3;
	//int verbose = 0;
	int _final  = 1;
	//cuhre.flags = verbose | 4;
	//cuhre.flags = verbose | 0;
	//cuhre.flags = 1;
	cuhre.maxeval = maxeval;
	double epsrel = 5.0e-3;
	double true_value = 0.;
	
	/*while(time_and_call_alt<cubacpp::Cuhre, SigmaMiscentY1ScalarIntegrand>(cuhre, integrand, epsrel, true_value, "dc_f0", 0)){
		epsrel = epsrel/1.5;
	}*/
	
	integral<GPU> d_integrand;
	d_integrand.set_grid_point({zo_low_, zo_high_, radius_});
    
	while(time_and_call<integral<GPU>>("pdc_f1_latest",
								 d_integrand,
								 epsrel,
								 0.,
							     "gpucuhre",
							     std::cout,
							     _final)){
						epsrel = epsrel/1.5;	
		break;
								 }
									 
	
	/*double bothEqual_cpu = integrand.mor->operator()(.6, 34., 0);
	std::cout<<"cpu_result:"<<bothEqual_cpu<<std::endl;
	
	mor_des_t<GPU> *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(mor_des_t<GPU>));
	memcpy(dhmf2, &d_integrand.mor, sizeof(mor_des_t<GPU>));
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, .6, 34., 0, gpu_result);
	cudaDeviceSynchronize();
	
	std::cout<<"gpu_result:"<<*gpu_result<<std::endl;
	if(*gpu_result == bothEqual_cpu)
		printf("Both equal Test Passed\n");
	double x_equal = integrand.mor->operator()(.6, 34.5, 0);
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, .6, 34.5, 0, gpu_result);
	cudaDeviceSynchronize();
	if(*gpu_result == x_equal)
		printf("X equal Test Passed\n");
	double y_equal = integrand.mor->operator()(.65, 34., 0);
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, .65, 34., 0, gpu_result);
	cudaDeviceSynchronize();
	if(*gpu_result == y_equal)
		printf("Y equal Test Passed\n");
	
	double x1_equal = integrand.mor->operator()(0.050000, 34., 0);
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, 0.050000, 34., 0, gpu_result);
	cudaDeviceSynchronize();
	if(*gpu_result == x1_equal)
		printf("x1_equal Test Passed\n");
	
	double y1_equal = integrand.mor->operator()(.6, .5, 0);
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, .6, .5, 0, gpu_result);
	cudaDeviceSynchronize();
	if(*gpu_result == y1_equal)
		printf("y1_equal Test Passed\n");
	else
		printf("y1_equal Test Failed\n");
	double xy1_equal = integrand.mor->operator()(0.050000, .5, 0);
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, 0.050000, .5, 0, gpu_result);
	cudaDeviceSynchronize();
	if(*gpu_result == xy1_equal)
		printf("xy1_equal Test Passed\n");
	
	double xy_last_equal = integrand.mor->operator()(2., 244., 0);
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, 2., 244., 0, gpu_result);
	cudaDeviceSynchronize();
	if(*gpu_result == xy_last_equal)
		printf("xy_last_equal Test Passed\n");
	
	double x_last_equal = integrand.mor->operator()(2., 243., 0);
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, 2., 243., 0, gpu_result);
	cudaDeviceSynchronize();
	if(*gpu_result == x_last_equal)
		printf("x_last_equal Test Passed\n");
	
	double y_last_equal = integrand.mor->operator()(1.9, 244., 0);
	testKernel<mor_des_t<GPU>><<<1,1>>>(dhmf2, 1.9, 244., 0, gpu_result);
	cudaDeviceSynchronize();
	if(*gpu_result == y_last_equal)
		printf("y_last_equal Test Passed\n");*/
	//printf("d_integrand Mor des cols:%lu\n", d_integrand.mor.sig_interp->_cols);
	//double cpu_result = d_integrand(lo, lc, lt, zt, lnM, rmis, theta);
	//CHECK(result == cpu_result);
	//kernel_wrapper<integral<GPU>>(d_integrand);
	//printf("repeat d_integrand Mor des cols:%lu\n", d_integrand.mor.sig_interp->_cols);
	
	//printf("After time and call\n");
	/*integral<GPU> *dhmf2;
	cudaMallocManaged((void**)&dhmf2, sizeof(integral<GPU>));
	memcpy(dhmf2, &d_integrand, sizeof(integral<GPU>));
	double* gpu_result;
	cudaMallocManaged((void**)&gpu_result, sizeof(double));
	printf("===============================\n");
	testKernel<integral<GPU>><<<1,1>>>(dhmf2, lo, lc, lt, zt, lnM, rmis, theta, gpu_result);
	cudaDeviceSynchronize();*/
	
	/*printf("-------------------------------\n");
	int _final = 0;
	printf("About to enter time and call loop\n");
	while (time_and_call<integral<GPU>>("pdc_f0_latest",
                       d_integrand,
                       epsrel,
                       true_value,
                       "gpucuhre",
                       std::cout,
                       _final) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
    }*/
  
  
    /*_final = 1;
	while (time_and_call<integral<GPU>>("pdc_f1_latest",
                       d_integrand,
                       epsrel,
                       true_value,
                       "gpucuhre",
                       std::cout,
                       _final) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
    }*/
	
	//printf("\n\ntrue:%.30f\n", result);
	//printf("cpu :%.30f\n", cpu_result);
	//printf("gpu :%.30f\n\n", *gpu_result);
	
	//printf("true:%e\n", result);
	//printf("cpu :%e\n", cpu_result);
	//printf("gpu :%e\n\n", *gpu_result);
	
	//printf("true:%a\n", result);
	//printf("cpu :%a\n", cpu_result);
	//printf("gpu :%a\n", *gpu_result);
	//CHECK(*gpu_result == cpu_result);
	 
	//cudaFree(dhmf2);
	//cudaFree(gpu_result);
//}
return 0;
}
