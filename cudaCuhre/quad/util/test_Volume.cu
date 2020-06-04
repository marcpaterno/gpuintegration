#include "catch2/catch.hpp"
#include "Volume.cuh"

using quad::Volume;

TEST_CASE("1d volume")
{
  SECTION("default constructed")
  {
    Volume<double, 1> vol;
    CHECK(vol.lows[0] == 0.0);
    CHECK(vol.highs[0] == 1.0);
  }
  SECTION("nondefault constructed")
  {
    double low = -5.0;
    double high = 10.5;
    Volume<double, 1> vol(&low, &high);
    CHECK(vol.lows[0] == low);
    CHECK(vol.highs[0] == high);
  }
};
