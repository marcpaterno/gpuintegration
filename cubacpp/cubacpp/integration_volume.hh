#ifndef CUBACPP_INTEGRATION_VOLUME_HH
#define CUBACPP_INTEGRATION_VOLUME_HH

#include "cubacpp/arity.hh"
#include "cubacpp/array.hh"

#include <array>
#include <functional>
#include <iterator>
#include <ostream>

namespace cubacpp {
  // Forward declaration of class template, needed for friend declarations.
  template <std::size_t N>
  class IntegrationVolume;

  // Declaration (but not definition) of friend function templates for
  // IntegrationVolume<N>.
  template <std::size_t N>
  std::ostream& operator<<(std::ostream&, IntegrationVolume<N> const&);
  template <std::size_t N>
  bool operator==(IntegrationVolume<N> const&, IntegrationVolume<N> const&);

  // IntegrationVolume<N> describes the a hyper-rectangular volume of
  // integration in N dimensions.
  template <std::size_t N>
  class IntegrationVolume {
    using array_t = array<N>;
    array_t lows_;
    array_t ranges_;
    double jacobian_;

  public:
    // The default integration volume is the unit hypercube.
    IntegrationVolume();

    // Create an integration volume in which the i'th variable ranges from
    // 'lows[i]' to 'highs[i]'. For each i, lows[i] must be less than highs[i].
    IntegrationVolume(array_t const& lows, array_t const& highs);

    // The jacobian of the transformation is the volume of the hyperrectangle.
    double jacobian() const;

    std::array<double, N> transform(std::array<double, N> const& in) const;

    // friend declarations for this N.
    friend std::ostream& operator<<<>(std::ostream& os,
                                      IntegrationVolume const& iv);
    friend bool operator==
      <>(IntegrationVolume const& a, IntegrationVolume const& b);
  };

  template <std::size_t N>
  std::ostream&
  operator<<(std::ostream& os, IntegrationVolume<N> const& iv)
  {
    os << "lows: ";
    for (int i = 0; i != N; ++i)
      os << iv.lows_[i] << ' ';
    os << "\nranges: ";
    for (int i = 0; i != N; ++i)
      os << iv.ranges_[i] << ' ';
    os << "\njacobian: " << iv.jacobian_ << '\n';
    return os;
  }

  template <std::size_t N>
  bool
  operator==(IntegrationVolume<N> const& a, IntegrationVolume<N> const& b)
  {
    return (a.jacobian_ == b.jacobian_) &&
           (a.lows_ == b.lows_) &&
           (a.ranges_ == b.ranges_);
  }

  template <std::size_t N>
  IntegrationVolume<N>::IntegrationVolume(array_t const& l, array_t const& h)
    : lows_{l}, ranges_{h - l}, jacobian_(ranges_.product())
  {}

  template <std::size_t N>
  IntegrationVolume<N>::IntegrationVolume() : lows_{}, ranges_{}, jacobian_(1)
  {
    // TODO: Find how Eigen can create these arrays more elegantly.
    lows_.fill(0.0);
    ranges_.fill(1.0);
  }

  template <std::size_t N>
  double
  IntegrationVolume<N>::jacobian() const
  {
    return jacobian_;
  }

  template <std::size_t N>
  std::array<double, N>
  IntegrationVolume<N>::transform(std::array<double, N> const& in) const
  {
    std::array<double, N> result;
    for (std::size_t i = 0; i != N; ++i) {
      result[i] = ranges_[i] * in[i] + lows_[i];
    }
    return result;
  }

  // for calculating the type of the integration volume suitable for use
  // with a callable object of type F.
  template <typename F>
  struct integration_volume_for {
    using type = IntegrationVolume<arity<F>()>;
  };

  // helper type, to reduce typing.
  template <typename F>
  using integration_volume_for_t = typename integration_volume_for<F>::type;
} // cubacpp

#endif
