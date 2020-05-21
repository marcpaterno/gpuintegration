#ifndef CUDACUHRE_QUAD_GPUQUAD_VOLUME_CUH
#define CUDACUHRE_QUAD_GPUQUAD_VOLUME_CUH

// user must make sure to call cudaMalloc and cudaMemcpy regarding d_highs and
// d_lows

template <typename T, int NDIM>
struct Volume {

  T* highs;
  T* lows;
  T* d_highs;
  T* d_lows;

  __host__
  Volume()
  {
    lows = new T[NDIM];
    highs = new T[NDIM];

    for (int i = 0; i < NDIM; i++) {
      lows[i] = 0;
      highs[i] = 1;
    }
  }

  __host__ __device__
  Volume(T* l, T* h, int dim)
  {
    lows = l;
    highs = h;
  }
};

#endif