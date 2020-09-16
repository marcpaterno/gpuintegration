#ifndef CUBACPP_GSL_HH
#define CUBACPP_GSL_HH

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h> // for gsl_function

#include "cubacpp/integrand_traits.hh"
#include "cubacpp/integration_result.hh"
#include <cstddef>
#include <memory>

namespace cubacpp {
  namespace detail {

    // IntegrationWorkspace is a resource manager for gsl_integration_workspace
    // objects. Constructing an IntegrationWorkspace allocates a
    // gsl_integration_workspace; destroying it deallocates the workspace.
    // Automatic conversion to gsl_integration_workspace* is provided.
    class IntegrationWorkspace {
    public:
      using workspace_t = gsl_integration_workspace;
      using dealloc_t = decltype(&gsl_integration_workspace_free);
      explicit IntegrationWorkspace(std::size_t n_subintervals);
      operator workspace_t*() const;

    private:
      std::unique_ptr<workspace_t, dealloc_t> _workspace;
    };

    inline IntegrationWorkspace::IntegrationWorkspace(
      std::size_t n_subintervals)
      : _workspace(gsl_integration_workspace_alloc(n_subintervals),
                   &gsl_integration_workspace_free)
    {}

    inline IntegrationWorkspace::operator workspace_t*() const
    {
      return _workspace.get();
    }

    // IntegrationCQUADWorkspace is a resource manager for
    // gsl_integration_cquad_workspace objects. Constructing an
    // IntegrationCQUADWorkspace allocates a gsl_integration_cquad_workspace;
    // destroying it deallocates the workspace.
    // Automatic conversion to gsl_integration_cquad_workspace* is provided.
    class IntegrationCQUADWorkspace {
    public:
      using workspace_t = gsl_integration_cquad_workspace;
      using dealloc_t = decltype(&gsl_integration_cquad_workspace_free);
      explicit IntegrationCQUADWorkspace(std::size_t n_subintervals);
      operator workspace_t*() const;

    private:
      std::unique_ptr<workspace_t, dealloc_t> _workspace;
    };

    inline IntegrationCQUADWorkspace::IntegrationCQUADWorkspace(
      std::size_t n_subintervals)
      : _workspace(gsl_integration_cquad_workspace_alloc(n_subintervals),
                   &gsl_integration_cquad_workspace_free)
    {}

    inline IntegrationCQUADWorkspace::operator workspace_t*() const
    {
      return _workspace.get();
    }

    // IntegrationQAWOTable is a resource manager for gsl_integration_qawo_table
    // objects. Constructing an IntegrationQAWOTable allocates a
    // gsl_integration_qawo_table; destroying it deallocates the table.
    // Automatic conversion to gsl_integration_qawo_table* is provided.
    class IntegrationQAWOTable {
    public:
      using table_t = gsl_integration_qawo_table;
      using dealloc_t = decltype(&gsl_integration_qawo_table_free);
      IntegrationQAWOTable(double omega,
                           double L,
                           gsl_integration_qawo_enum sin_or_cos,
                           std::size_t n_levels);
      operator table_t*() const;

    private:
      std::unique_ptr<table_t, dealloc_t> _table;
    };

    inline IntegrationQAWOTable::IntegrationQAWOTable(
      double omega,
      double L,
      gsl_integration_qawo_enum sin_or_cos,
      std::size_t n_levels)
      : _table(gsl_integration_qawo_table_alloc(omega, L, sin_or_cos, n_levels),
               &gsl_integration_qawo_table_free)
    {}

    inline IntegrationQAWOTable::operator table_t*() const
    {
      return _table.get();
    }

    // GSLIntegrator is the base type for all our GSL integration types that
    // integrate over finite ranges. Unlike CUBA, most GSL integration routines
    // integrate an explicit range (rather than across the unit hypercube). Ones
    // that inherit from GSLIntegrator will integrate from `range_start` to
    // `range_end`.
    struct GSLIntegrator {
      double range_start;
      double range_end;

      GSLIntegrator(double range_start, double range_end);

      // Helper member to ease changing integration ranges.
      // Note that with_range modifies the object on which it is called!
      GSLIntegrator& with_range(double start, double end);
    };

    inline GSLIntegrator::GSLIntegrator(double range_start, double range_end)
      : range_start(range_start), range_end(range_end)
    {}

    inline GSLIntegrator&
    GSLIntegrator::with_range(double start, double end)
    {
      range_start = start;
      range_end = end;
      return *this;
    }

    // GSLWorkspaceIntegrator is a GSLIntegrator that manages a
    // gsl_integration_workspace.
    struct GSLWorkspaceIntegrator : GSLIntegrator {
    protected:
      IntegrationWorkspace wkspc;

    public:
      std::size_t limit = 0;

      GSLWorkspaceIntegrator(double range_start,
                             double range_end,
                             std::size_t n)
        : GSLIntegrator(range_start, range_end), wkspc(n), limit(n)
      {}
    };

    // GSL takes a different integrand format from Cuba - this and the following
    // function will help that
    template <class F>
    double
    cuba_gsl_integrand(double x, void* params)
    {
      F* f = reinterpret_cast<F*>(params);
      return (*f)(x);
    }

    template <typename F>
    gsl_function
    make_gsl_integrand(F const* f)
    {
      using fn = integrand_traits<F>;
      static_assert(fn::ndim == 1, "GSL Integrals only support 1D");
      static_assert(
        std::is_same<typename fn::function_return_type, double>::value,
        "GSL Integrals only support scalar functions returning double");
      gsl_function out;
      out.function =
        static_cast<double (*)(double, void*)>(cuba_gsl_integrand<F>);
      out.params = const_cast<void*>(reinterpret_cast<void const*>(f));
      return out;
    }
  }

  // QNG - Quadrature, Non-Adaptive, General
  // Uses Gauss-Konrod rules to evaluate at a maximum of 81 points
  template <class F>
  integration_results<1>
  QNGIntegrate(F const& f, double a, double b, double epsrel, double epsabs)
  {
    // If key was not specified, deduce the highest order we can use based on
    // the dimensionality of the integrand.
    auto igrand = detail::make_gsl_integrand<F>(&f);
    double val, err;
    std::size_t neval;
    // Possible error codes:
    //  GSL_EMAXITER
    //  GSL_EROUND
    //  GSL_ESING
    //  GSL_EDIVERGE
    //  GSL_EDOM
    int retval =
      gsl_integration_qng(&igrand, a, b, epsabs, epsrel, &val, &err, &neval);
    // TODO: give sensible values for probability, neval, etc
    return {val,
            err,
            0.0,
            static_cast<long long>(neval),
            0,
            retval == GSL_SUCCESS ? 0 : 1};
  }

  // QAG - Quadrature, Adaptive, General
  template <class F>
  integration_results<1>
  QAGIntegrate(F const& f,
               gsl_integration_workspace* gsl_wkspc,
               double a,
               double b,
               double epsrel,
               double epsabs,
               int key = 1,
               std::size_t limit = 50000)
  {
    // If key was not specified, deduce the highest order we can use based on
    // the dimensionality of the integrand.
    auto igrand = detail::make_gsl_integrand<F>(&f);
    double val, err;
    // Possible error codes:
    //  GSL_EMAXITER
    //  GSL_EROUND
    //  GSL_ESING
    //  GSL_EDIVERGE
    //  GSL_EDOM
    int retval = gsl_integration_qag(
      &igrand, a, b, epsabs, epsrel, limit, key, gsl_wkspc, &val, &err);
    // TODO: give sensible values for probability, neval, etc
    return {val, err, 0.0, 0, 0, retval == GSL_SUCCESS ? 0 : 1};
  }

  template <class F>
  integration_results<1>
  CQUADIntegrate(F const& f,
                 gsl_integration_cquad_workspace* gsl_wkspc,
                 double a,
                 double b,
                 double epsrel,
                 double epsabs)
  {
    // If key was not specified, deduce the highest order we can use based on
    // the dimensionality of the integrand.
    auto igrand = detail::make_gsl_integrand<F>(&f);
    double val, err;
    std::size_t neval;
    // Possible error codes:
    //  GSL_EMAXITER
    //  GSL_EROUND
    //  GSL_ESING
    //  GSL_EDIVERGE
    //  GSL_EDOM
    int retval = gsl_integration_cquad(
      &igrand, a, b, epsabs, epsrel, gsl_wkspc, &val, &err, &neval);
    // TODO: give sensible values for probability, neval, etc
    return {
      val, err, 0.0, 0, static_cast<int>(neval), retval == GSL_SUCCESS ? 0 : 1};
  }

  // QNG - Quadrature, Non-Adaptive, General
  // See GSL docs:
  // https://www.gnu.org/software/gsl/doc/html/integration.html#c.gsl_integration_qng
  struct QNG : detail::GSLIntegrator {
    /* QNG will apply successive Gauss-Konrod rules of 10, 21, 43, or at most,
     * 87 points. If the error does not converge at 87 points, the integration
     * fails.
     */
    QNG(double range_start = 0, double range_end = 1)
      : GSLIntegrator(range_start, range_end)
    {}

    QNG&
    with_range(double start, double end)
    {
      GSLIntegrator::with_range(start, end);
      return *this;
    }

    template <class F>
    integration_results<1>
    integrate(F const& f, double epsrel, double epsabs) const
    {
      return QNGIntegrate(f, range_start, range_end, epsrel, epsabs);
    }
  };

  // QAG - Quadrature, Adaptive, General
  // See GSL docs:
  // https://www.gnu.org/software/gsl/doc/html/integration.html#cquad-doubly-adaptive-integration
  struct QAG : public detail::GSLWorkspaceIntegrator {
    int key = -1;

    /* The integration rule is determined by the value of key, which should be
     * chosen from the following symbolic names,
     *
     * GSL_INTEG_GAUSS15 (key = 1)
     * GSL_INTEG_GAUSS21 (key = 2)
     * GSL_INTEG_GAUSS31 (key = 3)
     * GSL_INTEG_GAUSS41 (key = 4)
     * GSL_INTEG_GAUSS51 (key = 5)
     * GSL_INTEG_GAUSS61 (key = 6)
     *
     * corresponding to the 15, 21, 31, 41, 51 and 61 point Gauss-Kronrod rules.
     */
    QAG(double range_start = 0,
        double range_end = 1,
        int key = GSL_INTEG_GAUSS61,
        std::size_t limit = 10)
      : GSLWorkspaceIntegrator(range_start, range_end, limit), key(key)
    {}

    QAG&
    with_range(double start, double end)
    {
      GSLIntegrator::with_range(start, end);
      return *this;
    }

    template <class F>
    integration_results<1>
    integrate(F const& f, double epsrel, double epsabs) const
    {
      return QAGIntegrate(
        f, wkspc, range_start, range_end, epsrel, epsabs, key, limit);
    }
  };

  // CQUAD doubly-adaptive integration
  // See GSL docs:
  // https://www.gnu.org/software/gsl/doc/html/integration.html#cquad-doubly-adaptive-integration
  struct CQUAD : detail::GSLIntegrator {
  protected:
    gsl_integration_cquad_workspace* wkspc;
    std::size_t nintervals;

  public:
    CQUAD(double range_start = 0.0, double range_end = 1.0, std::size_t n = 100)
      : GSLIntegrator(range_start, range_end)
      , wkspc(gsl_integration_cquad_workspace_alloc(n))
      , nintervals(n)
    {}

    CQUAD&
    with_range(double start, double end)
    {
      GSLIntegrator::with_range(start, end);
      return *this;
    }

    template <class F>
    integration_results<1>
    integrate(F const& f, double epsrel, double epsabs) const
    {
      return CQUADIntegrate(f, wkspc, range_start, range_end, epsrel, epsabs);
    }
  };

  template <typename F>
  integration_results<1>
  QAWFIntegrate(F const& f,
                gsl_integration_workspace* workspace,
                gsl_integration_workspace* cycle_workspace,
                gsl_integration_qawo_table* wf,
                double a,
                double epsabs,
                std::size_t n_intervals)
  {
    auto igrand = detail::make_gsl_integrand<F>(&f);
    double val, err;
    int retval = gsl_integration_qawf(&igrand,
                                      a,
                                      epsabs,
                                      n_intervals,
                                      workspace,
                                      cycle_workspace,
                                      wf,
                                      &val,
                                      &err);
    return {val, err, 0.0, 0, 0, retval == GSL_SUCCESS ? 0 : 1};
  }

  struct QAWF {
    detail::IntegrationWorkspace workspace;
    detail::IntegrationWorkspace cycle_workspace;
    detail::IntegrationQAWOTable wf;
    std::size_t n_intervals;
    double range_start; // QAWF has no range-end; it integrates to infinity.

    QAWF(std::size_t n_intervals,
         double start,
         double omega,
         gsl_integration_qawo_enum sin_or_cos,
         std::size_t n_levels);

    QAWF& with_range(double start);

    template <typename F>
    integration_results<1>
    integrate(F const& f, double epsabs) const
    {
      return QAWFIntegrate(
        f, workspace, cycle_workspace, wf, range_start, epsabs, n_intervals);
    }
  };

  inline QAWF::QAWF(std::size_t n_intervals,
                    double start,
                    double omega,
                    gsl_integration_qawo_enum sin_or_cos,
                    std::size_t n_levels)
    : workspace(n_intervals)
    , cycle_workspace(n_intervals)
    , wf(omega, 0.0, sin_or_cos, n_levels)
    , n_intervals(n_intervals)
    , range_start(start)
  {}

  inline QAWF&
  QAWF::with_range(double start)
  {
    range_start = start;
    return *this;
  }
}

#endif
