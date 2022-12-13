#include "cuda/cudaPagani/demos/demo_utils.cuh"
#include "cuda/cudaPagani/demos/function.cuh"
#include "cuda/cudaPagani/demos/compute_genz_integrals.cuh"
#include "cuda/mcubes/demos/demo_utils.cuh"
#include <string>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>

using namespace quad;

namespace detail {
 class Gaussian_2D {
  public:    
    __device__ __host__ double
    operator()(double x, double y)
    {
      return exp(
        -1.0 * (pow(x - 1., 2) + pow(y - 1., 2)));
    }
  };   
      
  class Gaussian_3D {
  public:    
    __device__ __host__ double
    operator()(double x, double y, double z)
    {
      return exp( -1.0 * ((pow(x - 1., 2) + pow(y - 1., 2) + pow(z - 1., 2))));
    }
  };  
    
  class Gaussian_4D {
  public:    
    __device__ __host__ double
    operator()(double x, double y, double z, double w)
    {
      return exp(
        -1.0 * (pow(x - 1., 2) + pow(y - 1., 2) +
                pow(z - 1., 2) + pow(w - 1., 2)));
    }
  };  
    
  class Gaussian_5D {
  public:    
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v)
    {
      return exp(
        -1.0 * (pow(x - 1., 2) + 
                pow(y - 1., 2) +
                pow(z - 1., 2) + 
                pow(w - 1., 2) +
                pow(v - 1., 2)));
    }
  };  
  
  class Gaussian_6D {
  public:    
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v, double t)
    {
      return exp(
        -1.0 * (pow(x - 1., 2) + 
                pow(y - 1., 2) +
                pow(z - 1., 2) + 
                pow(w - 1., 2) +
                pow(v - 1., 2)+
                pow(t - 1., 2)));
    }
  };
  
  class Gaussian_7D {
  public:    
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v, double t, double u)
    {
      return exp(
        -1.0 * (pow(x - 1., 2) + 
                pow(y - 1., 2) +
                pow(z - 1., 2) + 
                pow(w - 1., 2) +
                pow(v - 1., 2)+
                pow(t - 1., 2)+
                pow(u - 1., 2)));
    }
  };
  
  class Gaussian_8D {
  public:    
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v, double t, double u, double i)
    {
      return exp(
        -1.0 * (pow(x - 1., 2) + 
                pow(y - 1., 2) +
                pow(z - 1., 2) + 
                pow(w - 1., 2) +
                pow(v - 1., 2)+
                pow(t - 1., 2)+
                pow(u - 1., 2)+
                pow(i - 1., 2)));
    }
  };
  
  class Gaussian_9D {
  public:    
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v, double t, double u, double i, double p)
    {
      return exp(
        -1.0 * (pow(x - 1., 2) + 
                pow(y - 1., 2) +
                pow(z - 1., 2) + 
                pow(w - 1., 2) +
                pow(v - 1., 2)+
                pow(t - 1., 2)+
                pow(u - 1., 2)+
                pow(i - 1., 2)+
                pow(p - 1., 2)));
    }
  };
  
  class Gaussian_10D {
    public:    
    __device__ __host__ double
    operator()(double x, double y, double z, double w, double v, double t, double u, double i, double p, double k)
    {
      return exp(
        -1.0 * (pow(x - 1., 2) + 
                pow(y - 1., 2) +
                pow(z - 1., 2) + 
                pow(w - 1., 2) +
                pow(v - 1., 2)+
                pow(t - 1., 2)+
                pow(u - 1., 2)+
                pow(i - 1., 2)+
                pow(p - 1., 2)+
                pow(k - 1., 2)));
    }
  };
  
  
  class Oscillatory_2D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t)
  {
    return cos(s + t);
  }
};
  
  class Oscillatory_3D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t,
             double u)
  {
    return cos(s + t + u);
  }
};

  class Oscillatory_4D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v)
  {
    return cos(s + t + u + v);
  }
};

  class Oscillatory_5D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w)
  {
    return cos(s + t + u + v + w);
  }
};

  class Oscillatory_6D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x)
  {
    return cos(s + t + u + v + w + x);
  }
};
  
  class Oscillatory_7D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y)
  {
    return cos(s + t + u + v + w + x + y);
  }
};

  class Oscillatory_8D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z)
  {
    return cos(s + t + u + v + w + x + y + z);
  }
};
  
  class Oscillatory_9D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double c)
  {
    return cos(s + t + u + v + w + x + y + z + c);
  }
};

  class Oscillatory_10D {
    public:
    __device__ __host__ double
  operator()(double s,
             double t,
             double u,
             double v,
             double w,
             double x,
             double y,
             double z,
             double c,
             double b)
  {
    return cos(s + t + u + v + w + x + y + z + c + b);
  }
};
  
}

double random(double low, double high){
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(low, high);
    return distr(eng);
}

template<size_t ndim>
struct Range{
  double lows[ndim];
  double highs[ndim];
  
  Range(double low, double high){
    for(int dim = 0; dim < ndim; ++dim){
        lows[dim] = low;
        highs[dim] = high;
    }  
  }
};

template<typename IntegT, size_t ndim>
void multi_epsrel_integrate(const IntegT& integrand, Range<ndim> range, std::string label){
    quad::Volume<double, ndim> vol(range.lows, range.highs);
    Config configuration;
    std::array<double, 4> epsrels = {1.e-3, 1.e-4, 1.e-5, 1.e-6};
    double alpha = 1.;
    double beta = 1.;
    double true_value = scale(compute_gaussian<ndim>(alpha, beta), range.lows[0], range.highs[0]);
    
    for(auto epsrel : epsrels){
        cu_time_and_call<IntegT, ndim>("Gauss_" + std::to_string(ndim) + "D",
                                         integrand,
                                         epsrel,
                                         true_value,
                                         "gpucuhre",
                                         std::cout,
                                         configuration,
                                         &vol);
    }
}

void integrate_gaussians(){
    std::array<std::pair<double, double>, 5> ranges{{{0.,1.}, {-2., 2.}, {-3., 3.}, {-4., 4.}}};
        
    for(auto range : ranges){
        constexpr size_t ndim = 2;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_2D integrand;
        multi_epsrel_integrate<detail::Gaussian_2D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 3;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_3D integrand;
        multi_epsrel_integrate<detail::Gaussian_3D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 4;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_4D integrand;
        multi_epsrel_integrate<detail::Gaussian_4D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 5;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_5D integrand;
        multi_epsrel_integrate<detail::Gaussian_5D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 6;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_6D integrand;
        multi_epsrel_integrate<detail::Gaussian_6D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 7;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_7D integrand;
        multi_epsrel_integrate<detail::Gaussian_7D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 8;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_8D integrand;
        multi_epsrel_integrate<detail::Gaussian_8D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 9;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_9D integrand;
        multi_epsrel_integrate<detail::Gaussian_9D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 10;
        Range<ndim> test_range(range.first, range.second); 
        detail::Gaussian_10D integrand;
        multi_epsrel_integrate<detail::Gaussian_10D, ndim>(integrand, test_range, "Gauss");
        printf("-----------\n");
    }
}

void integrate_oscillatory(){
    std::array<std::pair<double, double>, 5> ranges{{{0.,1.}, {-2., 2.}, {-3., 3.}, {-4., 4.}}};
        
    for(auto range : ranges){
        constexpr size_t ndim = 2;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_2D integrand;
        multi_epsrel_integrate<detail::Oscillatory_2D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 3;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_3D integrand;
        multi_epsrel_integrate<detail::Oscillatory_3D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 4;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_4D integrand;
        multi_epsrel_integrate<detail::Oscillatory_4D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 5;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_5D integrand;
        multi_epsrel_integrate<detail::Oscillatory_5D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 6;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_6D integrand;
        multi_epsrel_integrate<detail::Oscillatory_6D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 7;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_7D integrand;
        multi_epsrel_integrate<detail::Oscillatory_7D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 8;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_8D integrand;
        multi_epsrel_integrate<detail::Oscillatory_8D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 9;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_9D integrand;
        multi_epsrel_integrate<detail::Oscillatory_9D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
    
    for(auto range : ranges){
        constexpr size_t ndim = 10;
        Range<ndim> test_range(range.first, range.second); 
        detail::Oscillatory_10D integrand;
        multi_epsrel_integrate<detail::Oscillatory_10D, ndim>(integrand, test_range, "Oscillatory");
        printf("-----------\n");
    }
}

int
main()
{
   integrate_gaussians();
   integrate_oscillatory();
}
