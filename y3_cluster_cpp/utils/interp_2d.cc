#include "interp_2d.hh"
#include "fpsupport.hh"

#include <algorithm>
#include <stdexcept>

y3_cluster::Interp2D::Interp2D(std::vector<Point3D>&& data)
  : xs_(), ys_(), zs_(), interp_(nullptr)
{
  make_grid_(std::move(data));
  interp_ = gsl_interp2d_alloc(gsl_interp2d_bilinear, nx(), ny());
  gsl_interp2d_init(interp_, xs_.data(), ys_.data(), zs_.data(), nx(), ny());
}

// below are added by Yuanyuan Zhang July 17
y3_cluster::Interp2D::Interp2D(std::vector<double>&& xs,
                               std::vector<double>&& ys,
                               std::vector<std::vector<double>> const& zs)
  : xs_(std::move(xs)), ys_(std::move(ys)), zs_(xs.size() * ys.size())
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

double
y3_cluster::Interp2D::operator()(double x, double y) const
{
  // We do not use the accelerator features of GSL interpolation, because we
  // do not expect that the pattern of calls will be such that it will help.
  // Profile the resulting integration routine to see if this should be changed.
  double z = 0.0;
  int rc = gsl_interp2d_eval_e(
    interp_, xs_.data(), ys_.data(), zs_.data(), x, y, nullptr, nullptr, &z);
  if (rc == 0)
    return z;
  std::cerr << "gsl error code: " << rc << '\n';
  std::cerr << "x: " << x << " y: " << y << '\n';
  std::abort();
  // Skip this check for now - we will fix this later
  /*
  double result = 0.0;
  int rc = gsl_interp2d_eval_e(
    interp_, xs_.data(), ys_.data(), zs_.data(), x, y, nullptr, nullptr,
  &result); if (rc == 0) return result;

  // We only get here on an error...
  std::cerr << "Failure in y3_cluster::Interp2D::operator()\n"
                << "x = " << x << " y = " << y << '\n';
  throw std::domain_error("argument out of range in Interp2D");
  */
}

y3_cluster::Interp2D::~Interp2D() noexcept
{
  gsl_interp2d_free(interp_);
}

void
y3_cluster::Interp2D::make_grid_(std::vector<Point3D>&& data)
{
  if (data.empty())
    throw std::domain_error("Interp2D -- no points provided");
  // Discover the (x,y) grid implied by the points we are given -- or reject
  // them as not a grid. Points in x are equivalent if they differ by an
  // absolute value less than xfuzz, and similarly for points in y.
  assure_clean_floats(data);

  // Sort first by y, then x, then z, within equivalence classes determined
  // by given relative and absolute tolerances. This is so that we have the
  // GSL-expected column-major ordering of the points.
  // TODO: Determine what tolerances are actually useful, and how they should
  // be exposed in the interface.
  double const reltol = 1.e-6;
  double const abstol = 1.e-24;
  Point3DLess comp{reltol, abstol, reltol, abstol};
  std::sort(data.begin(), data.end(), comp);

  // determine length of first block of points (how many x values for that y).
  // This is determining the value of M, the length of a column, which is the
  // number of rows.
  auto equal = [reltol, abstol, pa = data.begin()](Point3D const& b) {
    return fpsupport::is_equivalent((*pa)[1], b[1], reltol, abstol);
  };

  auto const column_1_end = std::find_if_not(data.begin(), data.end(), equal);
  if (column_1_end == data.end())
    throw std::domain_error("Interp2D -- Only one column");

  auto const candidate_nrows = std::distance(data.begin(), column_1_end);
  xs_.resize(candidate_nrows);

  // Capture the x values; if the constructor completes, these will have been
  // the correct values.
  for (std::size_t i = 0; i != static_cast<std::size_t>(candidate_nrows); ++i) {
    xs_[i] = data[i][0];
  }

  // Make sure the number of points is appropriate; determine the number of
  // columns.
  auto const candidate_ncolumns = data.size() / candidate_nrows;
  if (candidate_ncolumns * candidate_nrows != data.size())
    throw std::domain_error("Interp2D -- Not an integral number of columns");

  ys_.resize(candidate_ncolumns);

  // Make sure remaining columns conform: right number of rows, matching
  // x values. Capture the y values.
  auto const p_firstcolumn = data.cbegin();
  ys_[0] = (*p_firstcolumn)[1];

  auto equivalent_x_values = [reltol, abstol](Point3D const& a,
                                              Point3D const& b) {
    return fpsupport::is_equivalent(a[0], b[0], reltol, abstol);
  };
  for (std::size_t column = 1; column != candidate_ncolumns; ++column) {
    auto res = std::mismatch(p_firstcolumn,
                             p_firstcolumn + candidate_nrows,
                             p_firstcolumn + column * candidate_nrows,
                             equivalent_x_values);
    if (res.first != p_firstcolumn + candidate_nrows)
      throw std::domain_error("Interp2D -- Points do not form a grid");
    ys_[column] = (*(p_firstcolumn + column * candidate_nrows))[1]; // ugh
  }
  // Capture the z-matrix. Since the points are sorted, the order of the points
  // is the correct order for the zs_ vector.
  zs_.reserve(candidate_nrows * candidate_ncolumns);
  for (auto const& p : data)
    zs_.push_back(p[2]);
}
