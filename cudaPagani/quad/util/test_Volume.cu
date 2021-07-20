#define CATCH_CONFIG_MAIN
#include "cudaPagani/quad/util/Volume.cuh"
#include "catch2/catch.hpp"

using quad::Volume;

TEST_CASE("1d volume"){SECTION("default constructed"){Volume<double, 1> vol;
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
}
;

TEST_CASE("2d volume"){SECTION("default constructed"){Volume<double, 2> vol;
CHECK(vol.lows[0] == 0.0);
CHECK(vol.lows[1] == 0.0);
CHECK(vol.highs[0] == 1.0);
CHECK(vol.highs[1] == 1.0);
}
SECTION("nondefault constructed")
{
  double low[2] = {-5.0, 1.0};
  double high[2] = {10.5, 2.0};
  Volume<double, 2> vol(low, high);
  CHECK(vol.lows[0] == -5.0);
  CHECK(vol.highs[0] == 10.5);
  CHECK(vol.lows[1] == 1.0);
  CHECK(vol.highs[1] == 2.0);
}
}
;
