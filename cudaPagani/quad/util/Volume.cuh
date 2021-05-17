#ifndef CUDACUHRE_QUAD_GPUQUAD_VOLUME_CUH
#define CUDACUHRE_QUAD_GPUQUAD_VOLUME_CUH

#include <cstring>

// user must make sure to call cudaMalloc and cudaMemcpy regarding d_highs and
// d_lows

namespace quad {
  template <typename T, int NDIM>
  struct Volume {

    T lows[NDIM]  = {0.0};
    T highs[NDIM];

    __host__
    Volume()
    {
		for(T& x: highs)
			x = 1.0;
		
	}

    __host__ __device__
    Volume(T const* l, T const* h)
    {
      std::memcpy(lows, l, NDIM * sizeof(T));
      std::memcpy(highs, h, NDIM * sizeof(T));
    }

  };
}

#endif
