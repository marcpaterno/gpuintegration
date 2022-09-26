#ifndef ONE_API_QUAD_GPUQUAD_VOLUME_CUH
#define ONE_API_QUAD_GPUQUAD_VOLUME_CUH

#include <array>
#include <cstring>

// user must make sure to call cudaMalloc and cudaMemcpy regarding d_highs and
// d_lows

namespace quad {
  template <typename T, int NDIM>
  struct Volume {

    T lows[NDIM] = {0.0};
    T highs[NDIM];

    Volume()
    {
      for (T& x : highs)
        x = 1.0;
    }

    bool
    operator==(const Volume<double, NDIM>& vol)
    {
      for (int i = 0; i < NDIM; i++) {
        if (lows[i] != vol.lows[i])
          return false;
        if (highs[i] != vol.highs[i])
          return false;
      }
      return true;
    }

    bool
    operator!=(const Volume<double, NDIM>& vol)
    {

      for (int i = 0; i < NDIM; i++) {
        if (lows[i] == vol.lows[i])
          return true;
        if (highs[i] == vol.highs[i])
          return true;
      }
      return false;
    }

    Volume(std::array<T, NDIM> l, std::array<T, NDIM> h)
    {
      std::memcpy(lows, l.data(), NDIM * sizeof(T));
      std::memcpy(highs, h.data(), NDIM * sizeof(T));
    }

    Volume(T const* l, T const* h)
    {
      std::memcpy(lows, l, NDIM * sizeof(T));
      std::memcpy(highs, h, NDIM * sizeof(T));
    }
  };
}

#endif
