#ifndef CUBACPP_ARRAY_HH
#define CUBACPP_ARRAY_HH

namespace cubacpp {
  // We use std::size_t so that this template alias interacts well with sizeof,
  // etc. We use the static_cast because Eigen uses 'int' as the non-type
  // template parameter. Why, oh why?
  template <std::size_t N>
  struct array {
    void fill(double x);
    double operator[](std::size_t i) const;
    double& operator[](std::size_t i);
    array<N>& operator-=(array<N> const& other);
    double product() const;

    double data[N] = {0.0};
  };

  template <std::size_t N>
  void
  array<N>::fill(double x)
  {
    for (auto& v : data)
      v = x;
  }

  template <std::size_t N>
  double
  array<N>::operator[](std::size_t i) const
  {
    return data[i];
  }

  template <std::size_t N>
  double&
  array<N>::operator[](std::size_t i)
  {
    return data[i];
  }

  template <std::size_t N>
  array<N>&
  array<N>::operator-=(array<N> const& other)
  {
    for (std::size_t i = 0; i != N; ++i) {
      data[i] -= other[i];
    }
    return *this;
  }

  template <std::size_t N>
  double
  array<N>::product() const
  {
    double res = 1.0;
    for (auto x : data)
      res *= x;
    return res;
  }

  //--------------------------------------------------
  // related free functions

  template <std::size_t N>
  bool
  operator==(array<N> const& a, array<N> const& b)
  {
    // We could use std::equal, but this is so trivial...
    for (std::size_t i = 0; i != N; ++i) {
      if (a[i] != b[i])
        return false;
    }
    return true;
  }

  template <std::size_t N>
  array<N>
  operator-(array<N> const& a, array<N> const& b)
  {
    array<N> result(a);
    result -= b;
    return result;
  }
}

#endif
