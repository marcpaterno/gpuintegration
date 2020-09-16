#include "cubacpp/cuhre.hh"
#include "cubacpp/vegas.hh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

inline double
f2(double x, double y)
{
  return x * x + y + y + (x - y) * (x + y);
}

struct F2 {
  explicit F2(char const* fname) : out(std::make_shared<std::ofstream>(fname))
  {
    *out << "x\ty\n";
  };

  double
  operator()(double x, double y) const
  {
    *out << x << '\t' << y << '\n';
    return f2(x, y);
  };

  std::shared_ptr<std::ofstream> out;
};

int
main()
{
  cubacores(0, 0);
  F2 cuhre_f("cuhre_points.txt");
  F2 vegas_f("vegas_points.txt");

  cubacpp::Cuhre cuhre;
  cubacpp::Vegas vegas;
  auto rc = cuhre.integrate(cuhre_f, 1.e-2, 1.e-12);
  std::cout << "Did cuhre converge? " << (rc.status == 0 ? "yes" : "no")
            << '\n';
  std::cout << rc << '\n';

  auto rv = vegas.integrate(vegas_f, 1.e-2, 1.e-12);
  std::cout << "Did vegas converge? " << (rv.status == 0 ? "yes" : "no")
            << '\n';
  std::cout << rv << '\n';
}
