#ifndef Y3_CLUSTER_INTERP_1D_HH
#define Y3_CLUSTER_INTERP_1D_HH

#include "str_to_doubles.hh"
#include "gsl/gsl_interp.h"
#include <array>
#include <assert.h>
#include <cstddef>
#include <iomanip>
#include <istream>
#include <ostream>
#include <vector>

// Interp1D is used for linear interpolation in 1 dimension.
// It uses the GSL library to do the actual interpolation.
// Interp1D objects do not allow extrapolation.
//
namespace y3_cluster {
  class Interp1D {
  public:
    Interp1D() = default;

    // Interpolator created from two arrays; compiler assures they are of the
    // same length.
    template <std::size_t N>
    Interp1D(std::array<double, N> const& xs, std::array<double, N> const& ys);

    // Interpolator created from two vectors: throws std::logic_error if the
    // vectors are not of the same length.
    Interp1D(std::vector<double>&& xs, std::vector<double>&& ys);

    // As above, but deep-copy the vectors instead of just moving them. */
    Interp1D(std::vector<double> const& xs, std::vector<double> const& ys)
      : Interp1D{std::vector<double>(xs), std::vector<double>(ys)}
    {}

    // Destructor must clean up allocated GSL resources.
    ~Interp1D() noexcept;

    // Interp1D objects can not be copied because the GSL resources can not
    // be copied.
    Interp1D(Interp1D const&) = delete;

    // Ditto copy-assignment.
    Interp1D& operator=(Interp1D const&) = delete;

    double operator()(double x) const;

    double
    eval(double x) const
    {
      return this->operator()(x);
    };

    friend std::ostream&
    operator<<(std::ostream& os, Interp1D const& interp)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat;
      for (auto x : interp.xs_) {
        os << x << ' ';
      }
      os << '\n';
      for (auto y : interp.ys_) {
        os << y << ' ';
      }
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& in, Interp1D& interp)
    {
      assert(in.good());
      if (interp.interp_)
        gsl_interp_free(interp.interp_);
      std::string buffer;
      std::getline(in, buffer);
      interp.xs_ = cosmosis::str_to_doubles(buffer);
      assert(in.good());
      std::getline(in, buffer);
      interp.ys_ = cosmosis::str_to_doubles(buffer);
      interp.interp_ = gsl_interp_alloc(gsl_interp_linear, interp.xs_.size());
      gsl_interp_init(interp.interp_,
                      interp.xs_.data(),
                      interp.ys_.data(),
                      interp.xs_.size());
      return in;
    }

  private:
    std::vector<double> xs_;
    std::vector<double> ys_;
    gsl_interp* interp_ = nullptr;
  };
} // namespace y3_cluster

template <std::size_t N>
inline y3_cluster::Interp1D::Interp1D(std::array<double, N> const& xs,
                                      std::array<double, N> const& ys)
  : Interp1D({begin(xs), end(xs)}, {begin(ys), end(ys)})
{}

#endif
