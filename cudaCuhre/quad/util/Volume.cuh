#ifndef CUDACUHRE_QUAD_GPUQUAD_VOLUME_CUH
#define CUDACUHRE_QUAD_GPUQUAD_VOLUME_CUH

// user must make sure to call cudaMalloc and cudaMemcpy regarding d_highs and
// d_lows

template <typename T, int NDIM>
struct Volume {

  T highs[NDIM];
  T lows[NDIM];

  __host__
  Volume()
  {
	  for(int i=0; i<NDIM; i++){
		 highs[i]=1;
		 lows[i] =0;
	  }
  }

  __host__ __device__
  Volume(T const* l, T const* h)
  {
    memcpy(lows, l, NDIM * sizeof(T));
    memcpy(highs, h, NDIM * sizeof(T));
  }
};

#endif
