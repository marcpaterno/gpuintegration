#include "catch2/catch.hpp"
#include "modules/sigma_miscent_y1_scalarintegrand.hh"

#include <iostream>					//to overload >> in quad::Interp2D
#include "utils/str_to_doubles.hh"  //to utilize inside overloaded quad::Interp2D >> operator
#include <vector> 					

#include <fstream>
#include <stdexcept>
#include <string>

//using namespace y3_cluster;

namespace quad {
	
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
    __device__ __host__
    Interp2D(){};
	
    double* interpT;
    double* interpR;
    double* interpC;
    size_t _rows;
    size_t _cols;
	
    __host__ __device__
	Interp2D(double* xs, double* ys, double* zs, size_t cols, size_t rows){
		printf("Constructor called\n");
		//cudaMalloc((void**)&interpR, sizeof(double)*rows);
		//cudaMalloc((void**)&interpC, sizeof(double)*cols);
		//cudaMalloc((void**)&interpT, sizeof(double)*rows*cols);
		
		//cudaMemcpy(interpR, ys, sizeof(double)*rows, cudaMemcpyHostToDevice);
		//cudaMemcpy(interpC, xs, sizeof(double)*cols, cudaMemcpyHostToDevice);
		//cudaMemcpy(interpT, zs, sizeof(double)*rows*cols, cudaMemcpyHostToDevice);
		memcpy(interpR, ys, sizeof(double)*rows);
		memcpy(interpC, xs, sizeof(double)*cols);
		memcpy(interpT, zs, sizeof(double)*rows*cols);
		
		_rows = rows;
		_cols = cols;
    }
	
	__device__ __host__
	bool AreNeighbors(const double val, double* arr, const size_t leftIndex, const size_t RightIndex) const{
		if(arr[leftIndex] < val && arr[RightIndex] > val)
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
	  
	  interp._cols = xs.size();
	  interp._rows = ys.size();
	  cudaMallocManaged((void**)&(*&interp), sizeof(Interp2D));

	  cudaMallocManaged((void**)&interp.interpR, sizeof(double)*ys.size());
	  cudaDeviceSynchronize();
	  cudaMallocManaged((void**)&interp.interpC, sizeof(double)*xs.size());
	  cudaDeviceSynchronize();
	  cudaMallocManaged((void**)&interp.interpT, sizeof(double)*zs.size());
	  cudaDeviceSynchronize();
	  
	  memcpy(interp.interpR, ys.data(), sizeof(double)*ys.size());
	  memcpy(interp.interpC, xs.data(), sizeof(double)*xs.size());
	  memcpy(interp.interpT, zs.data(), sizeof(double)*zs.size());
	  
	  /*for(int i=0; i< interp._rows; i++)
		  printf("ys[%i]:%f\n", i, interp.interpR[i]);
	  for(int i=0; i< interp._cols; i++)
		  printf("xs[%i]:%f\n", i, interp.interpC[i]);
	  for(int i=0; i< zs.size(); i++)
		  printf("zs[%i]:%f\n", i, interp.interpT[i]);*/
	  
      return is;
    }
	
	__host__ __device__
	Interp2D(const Interp2D &source) {
		printf("Copy constructor called\n");
		interpT = source.interpT;
		interpC = source.interpC;
		interpR = source.interpR;
		_cols = source._cols;
		_rows = source._rows;
	} 
	
	//what to do if extrapolation is attempted?
	__device__ __host__
	void FindNeighbourIndices(const double val, double* arr, const size_t size, size_t& leftI, size_t& rightI) const{
		//assert for improper sizes?
		size_t currentIndex = size/2;
		size_t lastIndex = size - 1;
		leftI = 0;
		rightI = size - 1;
		
		//for(size_t i=0; i<size; ++i)
		//	printf("arr[%lu]:%f\n", i, arr[i]);
		
		while(currentIndex != 0 && currentIndex != lastIndex){
			currentIndex = leftI + (rightI - leftI)/2;
			//printf("currentIndex:%lu looking for %f within %lu range\n", currentIndex, val, size);
			if(AreNeighbors(val, arr, currentIndex-1, currentIndex)){
				leftI = currentIndex -1;
				rightI = currentIndex;
				return;
			}
			
			//printf("%f vs %f\n", arr[currentIndex], val);
			
			if(arr[currentIndex] > val){
				//printf("changing rightI from %lu to %lu\n", rightI, currentIndex);
				rightI = currentIndex;
			}
			else{
				//printf("changing leftI from %lu to %lu\n", leftI, currentIndex);
				leftI = currentIndex;
			}
			//currentIndex = arr[currentIndex] > val ? currentIndex /= 2 : currentIndex + (size-currentIndex)/2;
			//currentIndex = arr[currentIndex] > val ? (currentIndex-leftI) / 2 : currentIndex + (size-currentIndex)/2;
			
		}
		
		//values can't be found, how to handle?
		leftI  = 0;
		rightI = 0;
	}
	
    __device__ __host__ double
    operator()(double x, double y) const
    {
	  //y1, y2, x1, x2, are the indices of where to find the four neighbouring points in the z-table
	  size_t y1 = 0, y2 = 0;
	  size_t x1 = 0, x2 = 0;
		
	  FindNeighbourIndices(y, interpR, _rows, y1, y2);
	  FindNeighbourIndices(x, interpC, _cols, x1, x2);
	  
	  //printf("coordinates: %lu, %lu, %lu, %lu\n", y1, y2, x1, x2);
	  //this is how  zij is accessed by gsl2.6 Interp2D i.e. zij = z[j*xsize+i], where i=0,...,xsize-1, j=0, ..., ysize-1
	  const double q11 = interpT[y1*_cols + x1];
	  const double q12 = interpT[y2*_cols + x1];
	  const double q21 = interpT[y1*_cols + x2];
	  const double q22 = interpT[y2*_cols + x2];
	  //printf("values at coordinats:%.8f, %.8f, %.8f, %.8f\n", q11, q12, q21, q22);
	  
	  const double x1_val = interpC[x1];
	  const double x2_val = interpC[x2];
	  const double y1_val = interpR[y1];
	  const double y2_val = interpR[y2];
	  
	  const double f_x_y1 = q11*(x2_val-x)/(x2_val-x1) + q21*(x-x1_val)/(x2_val-x1_val);
	  const double f_x_y2 = q12*(x2_val-x)/(x2_val-x1_val) + q22*(x-x1_val)/(x2_val-x1_val);
	  
	  double f_x_y = f_x_y1*(y2_val-y)/(y2_val-y1_val) + f_x_y2*(y-y1_val)/(y2_val-y1_val); 
	  return f_x_y;
    }
	
	__device__ __host__ double
    min_x() const{ 
	return interpC[0]; }
	
	__device__ __host__ double
    max_x() const{ 
	return interpC[_cols-1];  }
	
    __device__  __host__  double
    min_y() const{ 
	return interpR[0]; }
	
    __device__ __host__ double
    max_y() const{ 
	return interpC[_rows-1]; }
	
	__device__  __host__ double
	do_clamp(double v, double lo, double hi) const
    {
		assert(!(hi < lo));
		return (v < lo) ? lo : (hi < v) ? hi : v;
    }
	
	__device__ __host__ double
    eval(double x, double y) const
    {
      return this->operator()(x, y);
    };
	
	__device__  __host__
    double
    clamp(double x, double y) const
    {
      return eval(do_clamp(x, min_x(), max_x()), do_clamp(y, min_y(), max_y()));
    }
  };
}

template <class T>
class hmf_t {
	  public:
	  
		__device__ __host__ 
		hmf_t() = default;
		__device__ __host__
		hmf_t(typename T::Interp2D* nmz, double s, double q)
		  : _nmz(nmz), _s(s), _q(q)
		{}
		
		using doubles = std::vector<double>;
		
		__device__ __host__
		double
		operator()(double lnM, double zt) const{
		  //printf("Inside operator ");
		  //printf("interpolation result:%f\n", _nmz->clamp(lnM, zt));
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
		  //auto table = std::make_shared<typename T::Interp2D>();
		  //needs to be deleted
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
		double _s = 0.0;
		double _q = 0.0;
};

struct GPU {
  typedef quad::Interp2D Interp2D;
};

struct CPU {
  typedef y3_cluster::Interp2D Interp2D;
};

template<typename T>
__global__ 
void
testKernel(T* model, double x, double y){
	//printf("Entered kernel\n");
	printf("quad:cudac gpu model:%.8f\n", model->operator()(x, y));
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

struct DataElement //: public Managed
{
  char *name;
  int value;
};

__global__ 
void Kernel(DataElement *elem) {
  printf("On device: name=%s, value=%d\n", elem->name, elem->value);

  elem->name[0] = 'd';
  elem->value++;
}

void launch(DataElement *elem) {
  Kernel<<< 1, 1 >>>(elem);
  cudaDeviceSynchronize();
}

class example{
	public:
		__host__ __device__
		example(){
			x = 1;
			y = 2;
			cudaMallocManaged((void**)&data, sizeof(int)*10);
			for(int i=0; i<10; i++){
				data[i] = 10+i;
				printf("%i\n", data[i]);
			}
		}
		
		int call(int i){
			printf("inside call\n");
			return data[0];}
		int x;
		int y; 
		int *data;
	
};

  template<class T>
  class X{
	public:
		hmf_t<T> model1;
		hmf_t<T> model2;
	    
		__device__ __host__
		double operator()(double x, double y){
			printf("model 1:%.8f\n", model1(x, y));
			printf("model 2:%.8f\n", model2(x, y));
		}
  };

int main(){
  hmf_t<CPU> hmf  = make_from_file<hmf_t<CPU>>("data/HMF_t.dump");
  hmf_t<GPU> hmf2 = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
  hmf_t<GPU> *dhmf2;
  cudaMallocManaged((void**)&dhmf2, sizeof(hmf_t<GPU>));
  cudaDeviceSynchronize();
  memcpy(dhmf2, &hmf2, sizeof(hmf_t<GPU>));
  X<GPU> x;
  x.model1 = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
  x.model2 = make_from_file<hmf_t<GPU>>("data/HMF_t.dump");
  double const zt = 0x1.cccccccccccccp-2;
  double const lnM = 0x1.0cp+5;
  x(lnM, zt);
  
  printf("y3_cluster cpu model:%.8f\n", hmf(lnM, zt));
  printf("quad:cudac cpu model:%.8f\n", dhmf2->operator()(lnM, zt));
  
  testKernel<hmf_t<GPU>><<<1,1>>>(dhmf2, lnM, zt);
  cudaDeviceSynchronize();
  
  return 0;
}
