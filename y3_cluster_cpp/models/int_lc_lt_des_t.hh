#ifndef Y3_CLUSTER_INT_LC_LT_DES_T_HH
#define Y3_CLUSTER_INT_LC_LT_DES_T_HH

#include "utils/datablock.hh"
#include "utils/interp_2d.hh"
#include "utils/primitives.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace y3_cluster {
  struct INT_LC_LT_DES_t {

    static Interp2D const lambda0_interp;
    static Interp2D const lambda1_interp;
    static Interp2D const lambda2_interp;
    static Interp2D const lambda3_interp;

    explicit INT_LC_LT_DES_t(const cosmosis::DataBlock&) {}
    INT_LC_LT_DES_t() {}

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
}

#endif
