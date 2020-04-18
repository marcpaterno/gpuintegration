#include "catch2/catch.hpp"
#include "models/angle_to_dist_t.hh"
#include "models/average_sci_t.hh"
#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/lo_lc_t.hh"
#include "models/mor_des_t.hh"
#include "models/mor_sdss_t.hh"
#include "models/roffset_t.hh"
#include "models/sig_sum.hh"
#include "utils/interp_1d.hh"

#include "utils/make_ifstream.hh"
#include "utils/str_to_doubles.hh"
#include <sstream>
#include <vector>

using std::hexfloat;
using std::ifstream;
using std::istringstream;
using std::ostringstream;
using std::stringstream;
using std::vector;

using namespace y3_cluster;

double const lo = 0x1.9p+4;
double const lc = 0x1.b8p+4;
double const lt = 0x1.b8p+4;
double const zt = 0x1.cccccccccccccp-2;
double const lnM = 0x1.0cp+5;
double const rmis = 0x1p+0;
// double const theta = 0x1.921fb54442eeap+1;
// double const radius_ = 0x1p+0;
// double const zo_low_ = 0x1.999999999999ap-3;
// double const zo_high_ = 0x1.6666666666666p-2;
double const scaled_Rmis = 0x0p+0;
// double const lc_lt = 0x1.733fa6defc7a2p-2;
double const mor = 0x1.e366843ef66fcp-6;
// double const omega_z = 0x1.d68f014e6c713p-2;
double const dv_do_dz = 0x1.84d005d9ad292p+31;
double const hmf = 0x1.3a3ec407b3769p-19;
// double const int_zo_zt = 0x1.cp-51;
double const roffset = 0x1.58fb9e70a6934p-3;
double const lo_lc = 0x1.4a4729d1398c6p-5;
double const sigma = 0x1.d6463be1b6a8cp+10;

TEST_CASE("Reading hexfloats")
{
  SECTION("one vector")
  {
    vector<double> expected{1.5, 2.5, 3.6};
    ostringstream out;
    out << hexfloat;
    for (auto x : expected) {
      out << x << ' ';
    };
    auto read = cosmosis::str_to_doubles(out.str());
    CHECK(expected == read);
  }
  SECTION("two vectors")
  {
    vector<double> first = {-1.5, 2.7e31};
    vector<double> second = {86, 99};
    ostringstream out;
    out << hexfloat;
    for (auto x : first) {
      out << x << ' ';
    };
    out << '\n';
    for (auto x : second) {
      out << x << ' ';
    };
    istringstream in(out.str());
    REQUIRE(in.good());
    std::string line;
    std::getline(in, line);
    CHECK(in.good());
    auto read_1 = cosmosis::str_to_doubles(line);
    CHECK(first == read_1);

    std::getline(in, line);
    CHECK(in.eof());
    auto read_2 = cosmosis::str_to_doubles(line);
    CHECK(second == read_2);
  }
}

TEST_CASE("Interp1D")
{
  SECTION("one table")
  {
    vector<double> xs = {1, 2, 3, 4};
    vector<double> ys = {2.5, 5.0, 7.5, 10.5};
    Interp1D m(xs, ys);
    CHECK(m(2.5) == 2.5 * 2.5);
    stringstream stream;
    stream << m;
    CHECK(stream.good());
    Interp1D t;
    stream >> t;
    CHECK(m(1.5) == t(1.5));
  }
  SECTION("table followed by double")
  {
    vector<double> xs = {1, 2, 3, 4};
    vector<double> ys = {2.5, 5.0, 7.5, 10.5};
    Interp1D m(xs, ys);
    CHECK(m(2.5) == 2.5 * 2.5);
    stringstream stream;
    stream << m;
    stream << '/';
    CHECK(stream.good());
    double d = 2.5;
    stream << d;
    CHECK(stream.good());
    Interp1D t;
    stream >> t;
    CHECK(m(1.5) == t(1.5));
    stream.clear();
    stream.ignore(2, '/');
    double val;
    stream >> val;
    CHECK(d == val);
  }
}

TEST_CASE("Interp2D")
{
  vector<double> xs = {1., 3.5, 7.};
  vector<double> ys = {10, 11, 12, 13};
  vector<double> zs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  Interp2D m(xs, ys, zs);
  stringstream stream;
  stream << m;
  CHECK(stream.good());
  Interp2D t;
  stream >> t;
  CHECK(m(2.4, 12.5) == t(2.4, 12.5));
}

TEST_CASE("HMF_t")
{
  ifstream is = make_ifstream("data/HMF_t.dump");
  REQUIRE(is.good());
  HMF_t m;
  is >> m;
  CHECK(m(lnM, zt) == hmf);
}

TEST_CASE("EZ_t")
{
  EZ const m(1.1, 2.2, 3.3);
  ostringstream out;
  out << m;
  CHECK(out.good());
  istringstream in(out.str());
  EZ t;
  in >> t;
  CHECK(m(2.3) == t(2.3));
}

TEST_CASE("EZ_sq")
{
  EZ_sq const m(1.1, 2.2, 3.3);
  ostringstream out;
  out << m;
  CHECK(out.good());
  istringstream in(out.str());
  EZ_sq t;
  in >> t;
  CHECK(m(2.3) == t(2.3));
}

TEST_CASE("DV_DO_DZ_t")
{
  SECTION("from string")
  {
    vector<double> const xs = {1, 2, 3};
    vector<double> const ys = {1.5, 2.5, 3.5};
    EZ const ez(1.1, 2.2, 3.3);
    double const h = 10.3;
    DV_DO_DZ_t m(std::make_shared<Interp1D>(xs, ys), ez, h);
    ostringstream out;
    out << m;
    istringstream in(out.str());
    CHECK(out.good());
    DV_DO_DZ_t t;
    in >> t;
    CHECK(m(2.5) == t(2.5));
  }
  SECTION("from file")
  {
    ifstream is = make_ifstream("data/DV_DO_DZ_t.dump");
    REQUIRE(is.good());
    DV_DO_DZ_t m;
    is >> m;
    CHECK(m(zt) == dv_do_dz);
  }
}

TEST_CASE("MOR_DES_t")
{
  ifstream is = make_ifstream("data/MOR_DES_t.dump");
  REQUIRE(is.good());
  MOR_DES_t m;
  is >> m;
  CHECK(m(lt, lnM, zt) == mor);
}

TEST_CASE("ANGLE_TO_DIST_t")
{
  vector<double> const xs = {0, 1, 2, 3};
  vector<double> const ys = {0, 1.5, 2.5, 3.5};
  double h_expected = 0.45;
  ANGLE_TO_DIST_t m(std::make_shared<Interp1D>(xs, ys), h_expected);
  ostringstream out;
  out << m;
  CHECK(out.good());
  istringstream in(out.str());
  ANGLE_TO_DIST_t t;
  in >> t;
  CHECK(m(1.1, 1.5) == t(1.1, 1.5));
}

TEST_CASE("ANGLE_TO_SCI_t")
{
  vector<double> const xs = {0, 1, 2, 3};
  vector<double> const ys = {0, 1.5, 2.5, 3.5};
  AVERAGE_SCI_t m(std::make_shared<Interp1D>(xs, ys));
  ostringstream out;
  out << m;
  CHECK(out.good());
  istringstream in(out.str());
  AVERAGE_SCI_t t;
  in >> t;
  CHECK(m(zt) == t(zt));
}

TEST_CASE("LO_LC_t.hh")
{
  ifstream in = make_ifstream("data/LO_LC_t.dump");
  REQUIRE(in.good());
  LO_LC_t m;
  in >> m;
  CHECK(m(lo, lc, rmis) == lo_lc);
}

TEST_CASE("MOR_sdss")
{
  MOR_sdss m(90124651837.9, 2291711130000.0, 0.698949129, 0.146305833);
  ostringstream out;
  out << m;
  REQUIRE(out.good());
  istringstream in(out.str());
  MOR_sdss t;
  in >> t;
  double const dummy = 0;
  CHECK(m(lt, lnM, dummy) == t(lt, lnM, dummy));
}

TEST_CASE("ROFFSET_t")
{
  ifstream is = make_ifstream("data/ROFFSET_t.dump");
  REQUIRE(is.good());
  ROFFSET_t m;
  is >> m;
  CHECK(m(rmis) == roffset);
}

TEST_CASE("SIG_SUM")
{
  ifstream is = make_ifstream("data/SIG_SUM.dump");
  REQUIRE(is.good());
  SIG_SUM m;
  is >> m;
  CHECK(m(scaled_Rmis, lnM, zt) == sigma);
}

// For the following classes, no i/o testing is needed. These
// classes fall into two classes:
//
// Classes for which each object is stateless.
//     INT_LC_LT_DES_t
//     LC_LT_t
//     OMEGA_Z_DES
//     OMEGA_Z_SDSS
//
// Classes for which each object has equivalent state.
//     INT_ZO_ZT_DES_t
