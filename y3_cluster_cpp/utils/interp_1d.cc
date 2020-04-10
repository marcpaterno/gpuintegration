#include "interp_1d.hh"
#include <stdexcept>

double
y3_cluster::Interp1D::operator()(double x) const
{
  double res;
  int rc = gsl_interp_eval_e(interp_, xs_.data(), ys_.data(), x, nullptr, &res);
  if (rc == 0)
    return res;
  throw std::domain_error("argument out of range in Interp1D");
}

y3_cluster::Interp1D::Interp1D(std::vector<double>&& xs,
                               std::vector<double>&& ys)
  : xs_(std::move(xs)), ys_(std::move(ys)), interp_(nullptr)
{
  if (xs_.size() != ys_.size())
    throw std::logic_error(
      "vector length mismatch in construction of Interp1D");

  interp_ = gsl_interp_alloc(gsl_interp_linear, xs_.size());
  gsl_interp_init(interp_, xs_.data(), ys_.data(), xs_.size());
}

y3_cluster::Interp1D::~Interp1D() noexcept
{
  gsl_interp_free(interp_);
}
