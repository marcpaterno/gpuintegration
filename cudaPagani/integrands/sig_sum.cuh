#ifndef SIG_SUM_CUH
#define SIG_SUM_CUH

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "y3_cluster_cpp/tests/cudaInterp.cuh"
namespace quad {

  template <class T>
  class sig_sum {
  private:
    // std::shared_ptr<typename T::Interp2D const> _sigma1;
    // std::shared_ptr<typename T::Interp2D const> _sigma2;
    // std::shared_ptr<typename T::Interp2D const> _bias;
    typename T::Interp2D* _sigma1;
    typename T::Interp2D* _sigma2;
    typename T::Interp2D* _bias;

  public:
    using doubles = std::vector<double>;
    sig_sum() = default;

    sig_sum(typename T::Interp2D* sigma1,
            typename T::Interp2D* sigma2,
            typename T::Interp2D* bias)
      : _sigma1(sigma1), _sigma2(sigma2), _bias(bias)
    {}

    ~sig_sum()
    {
      // just added
      // cudaFree(_sigma1);
      // cudaFree(_sigma2);
      // cudaFree(_bias);
      // delete _sigma1;
      // delete _sigma2;
      // delete _bias;
    }

    __host__ __device__ double
    operator()(double r, double lnM, double zt) const
    /*r in h^-1 Mpc */ /* M in h^-1 M_solar, represents M_{200} */
    {
      double _sig_1 = _sigma1->clamp(r, lnM);
      double _sig_2 = _bias->clamp(zt, lnM) * _sigma2->clamp(r, zt);
      // TODO: h factor?
      // if (del_sig_1 >= del_sig_2) {
      // return (1.+zt)*(1.+zt)*(1.+zt)*(_sig_1+_sig_2);
      return (_sig_1 + _sig_2);
      /*} else {
        return 1e12*(1.+zt)*(1.+zt)*(1.+zt)*del_sig_2;
      } */
    }

    friend std::ostream&
    operator<<(std::ostream& os, sig_sum const& m)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat << *m._sigma1 << '\n'
         << *m._sigma2 << '\n'
         << *m._bias;
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, sig_sum& m)
    {
      // auto sigma1 = std::make_shared<typename T::Interp2D>();
      typename T::Interp2D* sigma1 = new typename T::Interp2D;
      is >> *sigma1;
      if (!is)
        return is;
      // auto sigma2 = std::make_shared<typename T::Interp2D>();
      typename T::Interp2D* sigma2 = new typename T::Interp2D;
      is >> *sigma2;
      if (!is)
        return is;
      // auto bias = std::make_shared<typename T::Interp2D>();
      typename T::Interp2D* bias = new typename T::Interp2D;
      is >> *bias;
      if (!is)
        return is;
      m = sig_sum(sigma1, sigma2, bias);
      return is;
    }
  };
}
#endif