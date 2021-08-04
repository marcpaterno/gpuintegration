#include "cudaPagani/tests/model.hh"


namespace y3_cluster{


class EZ_sq {
  public:
    EZ_sq(double omega_m, double omega_l, double omega_k)
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

class EZ {
  public:
    EZ(double omega_m, double omega_l, double omega_k)
      : _ezsq(omega_m, omega_l, omega_k)
    {}

    double
    operator()(double z) const
    {
      auto const sqr = _ezsq(z);
      return std::sqrt(sqr);
    }

  private:
    EZ_sq _ezsq;
};

class DV_DO_DZ_t {
  public:
    DV_DO_DZ_t(y3_cluster::EZ ezt, double h)
      : _ezt(ezt), _h(h)
    {}

    double
    operator()(double zt) const
    {
      //double const da_z = _da(zt); // da_z needs to be in Mpc
      // Units: (Mpc/h)^3
      // 2997.92 is Hubble distance, c/H_0
      return 2997.92 * (1.0 + zt) * (1.0 + zt)  * _h  * _h /
             _ezt(zt);
    }

  private:
    //Interp1D _da; don't have Interp1D object
    y3_cluster::EZ _ezt;
    double _h;  
  };
}

template<typename Model, size_t arraySize>
std::array<double, arraySize>
Compute_CPU_model(const Model& model, const std::array<double, arraySize>& input){
    
    std::array<double, arraySize> output;
    
    for(size_t i = 0; i < arraySize; ++i)
        output[i] = model(input[i]);
    
    return output;
}

double* cpuExecute(){
    std::array<double, 10> zt_poitns = {0.156614, 0.239091, 0.3, 0.360909, 0.443386, 0.456614, 0.539091, 0.6, 0.660909, 0.743386};
    constexpr size_t arraySize = zt_poitns.size();
    std::array<double, 10> results;
    
    
    y3_cluster::EZ ezt(4.15, 3.13, 9.9);
    y3_cluster::DV_DO_DZ_t dv_do_dz_t(ezt, 7.4);
    
    results = Compute_CPU_model<y3_cluster::DV_DO_DZ_t, arraySize>(dv_do_dz_t, zt_poitns);
    for(auto i : results)
        std::cout<<i<<"\n";
    return results.data();
}
