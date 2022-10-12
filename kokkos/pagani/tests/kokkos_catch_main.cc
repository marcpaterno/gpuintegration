#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"

#include "Kokkos_Core.hpp"

int
main(int argc, char* argv[])
{
  int result = 0;
  Kokkos::initialize();
  {
    result = Catch::Session().run(argc, argv);
  }
  Kokkos::finalize();
  return result;
}
