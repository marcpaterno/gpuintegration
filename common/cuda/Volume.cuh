#ifndef CUDACUHRE_QUAD_GPUQUAD_VOLUME_CUH
#define CUDACUHRE_QUAD_GPUQUAD_VOLUME_CUH

#include <array>
#include <cstring>
#include <vector>

// user must make sure to call cudaMalloc and cudaMemcpy regarding d_highs and
// d_lows

namespace quad {
  template <typename T, int NDIM>
  struct Volume {

    T lows[NDIM] = {0.0};
    T highs[NDIM];

    __host__
    Volume()
    {
      for (T& x : highs)
        x = 1.0;
    }

    __host__
    Volume(std::array<T, NDIM> l, std::array<T, NDIM> h)
    {
      std::memcpy(lows, l.data(), NDIM * sizeof(T));
      std::memcpy(highs, h.data(), NDIM * sizeof(T));
    }

    __host__ __device__
    Volume(T const* l, T const* h)
    {
      std::memcpy(lows, l, NDIM * sizeof(T));
      std::memcpy(highs, h, NDIM * sizeof(T));
    }
	
	__host__
    Volume(T l, T h)
    {
	  std::array<T, NDIM> h_lows;	
	  std::array<T, NDIM> h_highs;
	  
	  std::fill(h_lows.begin(), h_lows.end(), l);
	  std::fill(h_highs.begin(), h_highs.end(), h);
		
      std::memcpy(lows, h_lows.data(), NDIM * sizeof(T));
      std::memcpy(highs, h_highs.data(), NDIM * sizeof(T));
    }
	
  };
}

#endif
