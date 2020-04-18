#ifndef Y3_CLUSTER_INTERP_2D_HH
#define Y3_CLUSTER_INTERP_2D_HH

#include "point_3d.hh"

#include "gsl/gsl_interp2d.h"
#include "utils/ndarray.hh"
#include "utils/str_to_doubles.hh"
#include <array>
#include <assert.h>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Interp2D is used for linear interpolation in 1 dimension.
// It uses the GSL library to do the actual interpolation.
// Interp2D object do not allow extrapolation.
//
namespace y3_cluster {

  // This is based on std::clamp<T> from C++ 17, put here for use with
  // models that use interpolation and need to avoid extrapolation.
  constexpr double
  do_clamp(double v, double lo, double hi)
  {
    assert(!(hi < lo));
    return (v < lo) ? lo : (hi < v) ? hi : v;
  }
  class Interp2D {
  public:
    // Interpolator created from two arrays; compiler assures they are of the
    // same length.
    template <std::size_t M, std::size_t N>
    using matrix = std::array<std::array<double, N>, M>;

    Interp2D() = default;

    template <std::size_t M, std::size_t N>
    Interp2D(std::array<double, M> const& xs,
             std::array<double, N> const& ys,
             matrix<M, N> const& zs);

    // Interpolator created from 3 vectors, specifying the x-axis, the y-axis,
    // and the column-major (N.B: not the natural-for-C++ row-major) storage of
    // the z values. We require the column-major ordering
    // because that is what is used by GSL.
    Interp2D(std::vector<double> const& xs,
             std::vector<double> const& ys,
             std::vector<double> const& zs);

    // Interpolator created from vector, vector, 2D vector, compiler assures
    // they are of the same length; Added by Yuanyuan Zhang
    Interp2D(std::vector<double>&& xs,
             std::vector<double>&& ys,
             std::vector<std::vector<double>> const& zs);

    // Interpolator created from vector, vector, 2D vector, compiler assures
    // they are of the same length; Added by Yuanyuan Zhang
    Interp2D(std::vector<double> const& xs,
             std::vector<double> const& ys,
             std::vector<std::vector<double>> const& zs);

    // Like above - assumes ndarray is 2D and fits it appropriately
    Interp2D(std::vector<double> const& xs,
             std::vector<double> const& ys,
             cosmosis::ndarray<double> const& zs)
      : Interp2D(xs, ys, std::vector<double>{zs.begin(), zs.end()})
    {
      if ((zs.extents()[1] != xs.size()) || (zs.extents()[0] != ys.size())) {
        std::cerr << "Interp2D -- wrong input dimensions:\n\t"
                  << "xs.size() = " << xs.size() << "\n\t"
                  << "ys.size() = " << ys.size() << "\n\t"
                  << "zs.shape[1] = " << zs.extents()[1] << "\n\t"
                  << "zs.shape[0] = " << zs.extents()[0] << "\n";
        throw std::domain_error("Interp2D -- ndarray wrong dimensions");
      }
    }

    // Interpolator created from vector of triplets: throws std::logic_error if
    // the points do not lie on a grid in (x,y) space, or if any values are NaN
    // or infinities. Any denormalized x- or y-values are flushed to zero.
    Interp2D(std::vector<Point3D>&& data);

    // As above, but take a deep copy of the argument for working space.
    Interp2D(std::vector<Point3D> const& data)
      : Interp2D(std::vector<Point3D>(data))
    {}

    // Destructor must clean up allocated GSL resources.
    ~Interp2D() noexcept;

    // Interp2D objects can not be copied because the GSL resources can not
    // be copied.
    Interp2D(Interp2D const&) = delete;

    // Ditto for the copy-assignment.
    Interp2D operator=(Interp2D const&) = delete;

    double operator()(double x, double y) const;

    double
    eval(double x, double y) const
    {
      return this->operator()(x, y);
    };

    double
    clamp(double x, double y) const
    {
      return eval(do_clamp(x, min_x(), max_x()), do_clamp(y, min_y(), max_y()));
    }

    // Return the number of grid points in x and y.
    std::size_t nx() const;
    std::size_t ny() const;

    friend std::ostream&
    operator<<(std::ostream& os, Interp2D const& interp)
    {
      auto const old_flags = os.flags();
      os << std::hexfloat;
      for (auto x : interp.xs_)
      {
        os << x << ' ';
      }
      os << '\n';
      for (auto y : interp.ys_)
    {
      os << y << ' ';
    }
      os << '\n';
      for (auto z : interp.zs_)
      {
        os << z << ' ';
      }
      os.flags(old_flags);
      return os;
    }

    friend std::istream&
    operator>>(std::istream& is, Interp2D& interp)
    {
      assert(is.good());
      if (interp.interp_)
        gsl_interp2d_free(interp.interp_);
      std::string buffer;
      std::getline(is, buffer);
      interp.xs_ = cosmosis::str_to_doubles(buffer);
      std::getline(is, buffer);
      interp.ys_ = cosmosis::str_to_doubles(buffer);
      std::getline(is, buffer);
      interp.zs_ = cosmosis::str_to_doubles(buffer);
      interp.interp_ =
        gsl_interp2d_alloc(gsl_interp2d_bilinear, interp.nx(), interp.ny());
      gsl_interp2d_init(interp.interp_,
                        interp.xs_.data(),
                        interp.ys_.data(),
                        interp.zs_.data(),
                        interp.nx(),
                        interp.ny());
      return is;
    }

  private:
    std::vector<double> xs_;
    std::vector<double> ys_;
    std::vector<double> zs_; // stores z-values in row-major
    gsl_interp2d* interp_ = nullptr;

    // Discover the (x,y) grid implicit in the supplied set of points, or throw
    // a std::domain_error if no grid can be constructed from these points.
    void make_grid_(std::vector<Point3D>&& data);

    // Get the limits of x and y.
    double
    min_x() const
    {
      return xs_.front();
    }
    double
    max_x() const
    {
      return xs_.back();
    }
    double
    min_y() const
    {
      return ys_.front();
    }
    double
    max_y() const
    {
      return ys_.back();
    }
  };
} // namespace y3_cluster

template <std::size_t M, std::size_t N>
inline y3_cluster::Interp2D::Interp2D(std::array<double, M> const& xs,
                                      std::array<double, N> const& ys,
                                      matrix<M, N> const& zs)
  : xs_(cbegin(xs), cend(xs)), ys_(cbegin(ys), cend(ys)), zs_(M * N)
{
  for (std::size_t i = 0; i != M; ++i) {
    std::array<double, N> const& row = zs[i];
    for (std::size_t j = 0; j != N; ++j) {
      zs_[i + j * M] = row[j];
    }
  }
  interp_ = gsl_interp2d_alloc(gsl_interp2d_bilinear, nx(), ny());
  gsl_interp2d_init(interp_, xs_.data(), ys_.data(), zs_.data(), nx(), ny());
}

// below are added by Yuanyuan Zhang July 17
inline y3_cluster::Interp2D::Interp2D(
  std::vector<double> const& xs,
  std::vector<double> const& ys,
  std::vector<std::vector<double>> const& zs)
  : xs_(xs), ys_(ys), zs_(xs.size() * ys.size())
{
  if (zs.size() != xs.size())
    throw std::domain_error("Interp2D -- wrong number of rows in z values");

  for (std::size_t i = 0; i < xs.size(); ++i) {
    std::vector<double> const& row = zs[i];
    if (row.size() != ys.size())
      throw std::domain_error(
        "Interp2D -- wrong number of columns in z values");
    for (std::size_t j = 0; j < ys.size(); ++j) {
      zs_[i + j * ys.size()] = row[j];
    }
  }

  if (zs_.size() != nx() * ny())
    throw std::domain_error("Interp2D -- wrong number of z values passed");
  interp_ = gsl_interp2d_alloc(gsl_interp2d_bilinear, nx(), ny());
  gsl_interp2d_init(interp_, xs_.data(), ys_.data(), zs_.data(), nx(), ny());
}

inline y3_cluster::Interp2D::Interp2D(std::vector<double> const& xs,
                                      std::vector<double> const& ys,
                                      std::vector<double> const& zs)
  : xs_(xs), ys_(ys), zs_(zs)
{
  if (zs_.size() != nx() * ny())
    throw std::domain_error("Interp2D -- wrong number of z values passed");
  interp_ = gsl_interp2d_alloc(gsl_interp2d_bilinear, nx(), ny());
  gsl_interp2d_init(interp_, xs_.data(), ys_.data(), zs_.data(), nx(), ny());
}

inline std::size_t
y3_cluster::Interp2D::nx() const
{
  return xs_.size();
}

inline std::size_t
y3_cluster::Interp2D::ny() const
{
  return ys_.size();
}

#endif
