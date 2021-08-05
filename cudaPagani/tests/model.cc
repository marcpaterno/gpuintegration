#include "cudaPagani/tests/model.hh"

namespace y3_cluster {

  class EZ_sq_simplified {
  public:
    EZ_sq_simplified(double omega_m, double omega_l, double omega_k)
      : _omega_m(omega_m), _omega_l(omega_l), _omega_k(omega_k)
    {}
    double
    operator()(double z) const
    {
      // NOTE: this is valid only for \Lambda CDM cosmology, not wCDM
      double const zplus1 = 1.0 + z;
      return (_omega_m * zplus1 * zplus1 * zplus1 + _omega_k * zplus1 * zplus1 +
              _omega_l);
    }

  private:
    double _omega_m;
    double _omega_l;
    double _omega_k;
  };

  class EZ_simplified {
  public:
    EZ_simplified(double omega_m, double omega_l, double omega_k)
      : _ezsq(omega_m, omega_l, omega_k)
    {}

    double
    operator()(double z) const
    {
      auto const sqr = _ezsq(z);
      return std::sqrt(sqr);
    }

  private:
    EZ_sq_simplified _ezsq;
  };

  class DV_DO_DZ_t_simplified {
  public:
    DV_DO_DZ_t_simplified(y3_cluster::EZ_simplified ezt, double h)
      : _ezt(ezt), _h(h)
    {}

    double
    operator()(double zt) const
    {
      // double const da_z = _da(zt); // da_z needs to be in Mpc
      // Units: (Mpc/h)^3
      // 2997.92 is Hubble distance, c/H_0
      // return 2997.92 * (1.0 + zt) * (1.0 + zt)  * sqrt(_h)  * erf(_h) /
      // _ezt(zt);
      return 2997.92 * (1.0 + zt) * (1.0 + zt) * (_h) * (_h) / _ezt(zt);
    }

  private:
    // Interp1D _da; don't have Interp1D object
    y3_cluster::EZ_simplified _ezt;
    double _h;
  };
}

template <typename Model>
std::vector<double>
Compute_CPU_model(const Model& model, const std::vector<double> input)
{

  std::vector<double> output;
  output.reserve(input.size());

  for (size_t i = 0; i < input.size(); ++i)
    output.push_back(model(input[i]));

  return output;
}

std::vector<double>
cpuExecute()
{
  std::vector<double> zt_poitns = {0.156614,
                                   0.239091,
                                   0.3,
                                   0.360909,
                                   0.443386,
                                   0.456614,
                                   0.539091,
                                   0.6,
                                   0.660909,
                                   0.743386};
  std::vector<double> results;

  y3_cluster::EZ_simplified ezt(4.15, 3.13, 9.9);
  y3_cluster::DV_DO_DZ_t_simplified dv_do_dz_t(ezt, 7.4);

  results =
    Compute_CPU_model<y3_cluster::DV_DO_DZ_t_simplified>(dv_do_dz_t, zt_poitns);
  return results;
}
