#include <iostream>
#include <Kokkos_Core.hpp>
#include "kokkos/pagani/demos/demo_utils.cuh"

template<bool Regular>
class F_3_2D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
               double y)
  {
	if constexpr(Regular == true)  
		return pow(1. + x + 2. * y, -3.);
	else
		return 1.;
  }
};

template<bool Regular>
class F_3_3D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
               double y,
               double z)
  {
	if constexpr(Regular == true)  
		return pow(1. + 3. * x + 2. * y + z, -4.);
	else
		return 1.;
  }
};

template<bool Regular>
class F_3_4D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
               double y,
               double z,
               double w)
  {
	if constexpr(Regular == true)    
		return pow(1. + 4. * w + 3. * x + 2. * y + z, -5.);
	else
		return 1.;
  }
};

template<bool Regular>
class F_3_5D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
               double y,
               double z,
               double w,
               double v)
  {
	if constexpr(Regular == true)    
		return pow(1. + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
	else
		return 1.;
  }
};

template<bool Regular>
class F_3_6D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u)
  {
	if constexpr(Regular == true)    
		return pow(1. + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -7.);
	else
		return 1.;
  }
};

template<bool Regular>
class F_3_7D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t)
  {
	if constexpr(Regular == true)    
		return pow(1. + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -8.);
	else
		return 1.;
  }
};

template<bool Regular>
class F_3_8D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x,
               double y,
               double z,
               double w,
               double v,
               double u,
               double t,
               double s)
  {
	if constexpr(Regular == true)    
		return pow(1. + 8. * s + 7. * t + 6. * u + 5. * v + 4. * w + 3. * x + 2. * y + z, -9.);
	else
		return 1.;
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
  call_cubature_rules<F_3_8D<true>, 8>(num_repeats);
  call_cubature_rules<F_3_7D<true>, 7>(num_repeats);
  call_cubature_rules<F_3_6D<true>, 6>(num_repeats);
  call_cubature_rules<F_3_5D<true>, 5>(num_repeats);
  call_cubature_rules<F_3_4D<true>, 4>(num_repeats);
  call_cubature_rules<F_3_3D<true>, 3>(num_repeats);
  call_cubature_rules<F_3_2D<true>, 2>(num_repeats);

  call_cubature_rules<F_3_8D<false>, 8>(num_repeats);
  call_cubature_rules<F_3_7D<false>, 7>(num_repeats);
  call_cubature_rules<F_3_6D<false>, 6>(num_repeats);
  call_cubature_rules<F_3_5D<false>, 5>(num_repeats);
  call_cubature_rules<F_3_4D<false>, 4>(num_repeats);
  call_cubature_rules<F_3_3D<false>, 3>(num_repeats);
  call_cubature_rules<F_3_2D<false>, 2>(num_repeats);
  Kokkos::finalize();
  return 0;
}
