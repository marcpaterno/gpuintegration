#ifndef HMF_T_CUH
#define HMF_T_CUH

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "quad/GPUquad/Interp2D.cuh"

namespace quad {

  template <class T>
  class hmf_t {
  public:
    hmf_t() = default;

    __host__
    hmf_t(typename T::Interp2D* nmz, double s, double q)
      : _nmz(nmz), _s(s), _q(q)
    {}

    using doubles = std::vector<double>;

    // ADD DATABLOCK CONSTRUCTOR
    __device__ __host__ double
    operator()(double lnM, double zt) const
    {
      return _nmz->clamp(lnM, zt) *
             (_s * (lnM * 0.4342944819 - 13.8124426028) + _q);
    }

    friend std::ostream&
    operator<<(std::ostream& os, hmf_t const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat;
      os << *(m._nmz) << '\n' << m._s << ' ' << m._q;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, hmf_t& m)
    {
      assert(is.good());
      // doing the line below instead //auto table = std::make_shared<typename
      // T::Interp2D>();
      typename T::Interp2D* table = new typename T::Interp2D;
      is >> *table;

      std::string buffer;
      std::getline(is, buffer);
      std::vector<double> const vals_read = str_to_doubles(buffer);
      if (vals_read.size() == 2) {
        m = hmf_t(table, vals_read[0], vals_read[1]);
      } else {
        is.setstate(std::ios_base::failbit);
      };
      return is;
    }

  private:
    typename T::Interp2D* _nmz;
    // std::shared_ptr<typename T::Interp2D const> _nmz;

    double _s = 0.0;
    double _q = 0.0;
  };
}
#endif