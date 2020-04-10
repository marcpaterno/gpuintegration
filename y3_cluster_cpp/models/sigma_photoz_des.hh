#ifndef Y3_CLUSTER_SIGMA_PHOTOZ_T_HH
#define Y3_CLUSTER_SIGMA_PHOTOZ_T_HH

#include <cmath>

namespace y3_cluster {

  class SIGMA_PHOTOZ_DES_t {
  public:
    explicit SIGMA_PHOTOZ_DES_t() {}

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
}

#endif
