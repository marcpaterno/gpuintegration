#ifndef Y3_CLUSTER_POINT_3D_HH
#define Y3_CLUSTER_POINT_3D_HH

#include "fpsupport.hh"

#include <array>
#include <vector>

namespace y3_cluster {

  // Point3D represents an (x,y,z) triplet.
  using Point3D = std::array<double, 3>;

  inline bool
  icky(y3_cluster::Point3D const& p)
  {
    using fpsupport::icky;
    return icky(p[0]) || icky(p[1]) || icky(p[2]);
  }

  // Replace any subnormal x- or y-values by 0. We assume there are no NaN or
  // infinite values in p.
  inline void
  squash_subnormals(y3_cluster::Point3D& p)
  {
    if (not std::isnormal(p[0]))
      p[0] = 0.0;
    if (not std::isnormal(p[1]))
      p[1] = 0.0;
  }

  // "Clean" the input points. If any NaN or infinity values are detected,
  // throw std::domain_error. Replace any x- or y-value denormals by zero.
  inline void
  assure_clean_floats(std::vector<y3_cluster::Point3D>& points)
  {
    for (auto& p : points) {
      if (icky(p))
        throw std::domain_error("Inf. or NaN detected in Interp2D setup");
      squash_subnormals(p);
    }
  }

  // Point3DLess is a predicate object for ordering Point3D objects, in a manner
  // suitable for the establishment of (x,y) grids for interpolation in 2D. For
  // identifying the x- and y-coordinates of the grid, approximate equality
  // testing is used. 'xabs' and 'xrel' are the absolute and relative tolerances
  // for evaluating the equality of the x-coordinates, and 'yabs' and 'yrel'
  // those for the y-coordinates. We sort into column-major order, to satisfy
  // GSL.
  class Point3DLess {
  public:
    Point3DLess(double xrel, double xabs, double yrel, double yabs) noexcept;
    bool operator()(Point3D const& a, Point3D const& b) const;

  private:
    double xrel_;
    double xabs_;
    double yrel_;
    double yabs_;
  };
} // namespace y3_cluster

inline y3_cluster::Point3DLess::Point3DLess(double xrel,
                                            double xabs,
                                            double yrel,
                                            double yabs) noexcept
  : xrel_(xrel), xabs_(xabs), yrel_(yrel), yabs_(yabs)
{}

inline bool
y3_cluster::Point3DLess::operator()(Point3D const& a, Point3D const& b) const
{
  if (not fpsupport::is_equivalent(a[1], b[1], yabs_, yrel_))
    return a[1] < b[1];
  if (not fpsupport::is_equivalent(a[0], b[0], xabs_, xrel_))
    return a[0] < b[0];
  return a[2] < b[2];
}

#endif
