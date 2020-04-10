#ifndef Y3_CLUSTER_CPP_MAKE_GRID_POINTS_HH
#define Y3_CLUSTER_CPP_MAKE_GRID_POINTS_HH

#include "utils/datablock.hh"
#include "utils/datablock_reader.hh"
#include "utils/meta.hh"

#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple> // for std::apply
#include <vector>

namespace y3_cluster {
  // These variadic function templates takes as arguments:
  //   1. a cosmosis::DataBlock (by reference),
  //   2. the name of the module being configured, and
  //   3. one or more arguments that are convertible to strings.
  //
  // Assuming your class's name is ClsName, it should be used like:
  //    DataBlock cfg;
  //    auto vols = make_grid_points(cfg, ClsName::module_label(), "radius",
  //    "z");
  //
  // The function returns a vector of std::array<double, N>,
  // where N is the number of names provided.
  //
  // make_grid_points_cartesian_product expects one configuration
  // parameter for each named axis, and forms a grid from the
  // cartesian product of the axis values.
  //
  // make_grid_points_wall_of_numbers expects one configuration
  // parameter for each named axis. All must be vectors of the
  // same length. The resulting grid points carry those axis
  // values.
  template <typename... Ts>
  std::vector<std::array<double, sizeof...(Ts)>>
  make_grid_points_cartesian_product(cosmosis::DataBlock& cfg,
                                     std::string const& modulelabel,
                                     Ts... stringlikes);

  template <typename... Ts>
  std::vector<std::array<double, sizeof...(Ts)>>
  make_grid_points_wall_of_numbers(cosmosis::DataBlock& cfg,
                                   std::string const& modulelabel,
                                   Ts... names);

  template <typename... Ts>
  std::vector<std::array<double, sizeof...(Ts)>>
  load_grid_from_file_wall_of_numbers(cosmosis::DataBlock& cfg,
                                      std::string const& modulelabel,
                                      Ts... names);

  namespace detail {

    // Given a vector of N names of configuration parameters, look up
    // the N vectors of grid locations for each of the Nm axes, and
    // put them into 'axes'.
    template <std::size_t N>
    void
    get_grid_axes_from_datablock(cosmosis::DataBlock& cfg,
                                 std::string const& modulelabel,
                                 std::array<std::string, N> const& names,
                                 std::array<std::vector<double>, N>& axes)
    {
      auto get_axis = [&cfg, &modulelabel](std::string const& name) {
        return get_vector_double(cfg, modulelabel.c_str(), name.c_str());
      };
      std::transform(names.begin(), names.end(), axes.begin(), get_axis);
    }

    // Given a filename in the configuration parameters, open the file,
    // parse each row into a grid point, and put them into 'axes'
    template <std::size_t N>
    void
    get_grid_axes_from_file(cosmosis::DataBlock& cfg,
                            std::string const& modulelabel,
                            std::array<std::vector<double>, N>& axes)
    {
      // Try to open the file
      std::string const gridfilename =
        cfg.view<std::string>(modulelabel, "grid_file");
      std::ifstream file(gridfilename);
      if (!file) {
        std::string errmsg("Failed to open file: ");
        errmsg += gridfilename;
        throw std::runtime_error(errmsg);
      }

      // Load the file
      std::string line;
      std::vector<double> onerow;
      std::vector<std::vector<double>> fullres;
      while (std::getline(file, line)) {
        // Skip comment lines
        if (line.find('#') == 0)
          continue;

        // Make the line into a vector of doubles
        std::istringstream linestream(line);
        double tmp;
        while (linestream >> tmp)
          onerow.push_back(tmp);
        fullres.push_back(onerow);
        onerow.clear();
      }

      // Place each column into the axes object
      std::size_t const n = fullres.size();
      std::vector<double> col;
      for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < n; ++j)
          col.push_back(fullres[j][i]);
        axes[i] = col;
        col.clear();
      }
    }

    // make_grid_splatted takes N vectors of floating point numbers, where N is
    // any positive integer, and returns a vector of grid points, where each
    // grid point is an array of N doubles.
    template <typename... Ts>
    std::vector<std::array<double, sizeof...(Ts)>>
    make_grid_splatted(std::vector<Ts> const&... axes)
    {
      // Make sure all the vectors are carrying floating point numbers;
      // we convert everything to double, 'cause doing otherwise is hard.
      static_assert(std::conjunction_v<std::is_floating_point<Ts>...>);

      std::vector<std::array<double, sizeof...(Ts)>> res;
      auto accumulator = [&res](Ts const&... ts) {
        // Construct an array from all the elements we pass in ts...
        res.push_back({ts...});
      };
      detail::cartesian_product(accumulator, axes...);
      return res;
    }

    // make_grid_aux is called with an array of arbitrary length, and an
    // index_sequence of matching length.
    template <std::size_t... Is>
    std::vector<std::array<double, sizeof...(Is)>>
    make_grid_aux(std::array<std::vector<double>, sizeof...(Is)> const& axes,
                  std::index_sequence<Is...> /*unused*/)
    {
      return make_grid_splatted(axes[Is]...);
    }

    template <std::size_t N>
    std::vector<std::array<double, N>>
    make_grid_cartesian_product(std::array<std::vector<double>, N> const& axes)
    {
      return detail::make_grid_aux(axes, std::make_index_sequence<N>());
    }

    template <std::size_t N>
    std::vector<std::array<double, N>>
    make_grid_wall_of_numbers(std::array<std::vector<double>, N> const& axes)
    {
      // Make sure all vectors have the same length
      std::size_t const n = axes[0].size();
      for (std::size_t i = 1; i < N; ++i) {
        if (axes[i].size() != n) {
          throw std::invalid_argument(
            "unequal length axes in make_grid_wall_of_numbers");
        }
      }
      std::vector<std::array<double, N>> res(n);
      for (std::size_t i = 0; i != n; ++i) {
        std::array<double, N>& current_result = res[i];
        for (std::size_t j = 0; j != N; ++j) {
          current_result[j] = axes[j][i];
        }
      }
      return res;
    }
  } // namespace detail
} // namespace y3_cluster

template <typename... STRINGLIKEs>
std::vector<std::array<double, sizeof...(STRINGLIKEs)>>
y3_cluster::make_grid_points_cartesian_product(cosmosis::DataBlock& cfg,
                                               std::string const& modulelabel,
                                               STRINGLIKEs... names)
{
  // Make sure that all arguments are convertible to std::string.
  static_assert(
    std::conjunction_v<std::is_convertible<STRINGLIKEs, std::string>...>,
    "\n\nCosmoSIS error!\nAll trailing arguments in "
    "make_grid_points_cartesian_product must be convertible to string.\n\n");
  constexpr std::size_t n_axes = sizeof...(STRINGLIKEs);

  std::array<std::string, n_axes> const axis_names{
    std::forward<STRINGLIKEs>(names)...};
  std::array<std::vector<double>, n_axes> axes;
  detail::get_grid_axes_from_datablock(cfg, modulelabel, axis_names, axes);
  return detail::make_grid_cartesian_product(axes);
}

template <typename... STRINGLIKEs>
std::vector<std::array<double, sizeof...(STRINGLIKEs)>>
y3_cluster::make_grid_points_wall_of_numbers(cosmosis::DataBlock& cfg,
                                             std::string const& modulelabel,
                                             STRINGLIKEs... names)
{
  // Make sure all the arguments are convertible to std::string.
  static_assert(
    std::conjunction_v<std::is_convertible<STRINGLIKEs, std::string>...>,
    "\n\nCosmoSIS error!\nAll trailing arguments in "
    "make_grid_points_wall_of_numbers must be convertible to string.\n\n");
  constexpr std::size_t n_axes = sizeof...(STRINGLIKEs);
  std::array<std::string, n_axes> const axis_names{
    std::forward<STRINGLIKEs>(names)...};
  std::array<std::vector<double>, n_axes> axes;
  detail::get_grid_axes_from_datablock(cfg, modulelabel, axis_names, axes);
  return detail::make_grid_wall_of_numbers(axes);
}

template <typename... STRINGLIKEs>
std::vector<std::array<double, sizeof...(STRINGLIKEs)>>
y3_cluster::load_grid_from_file_wall_of_numbers(cosmosis::DataBlock& cfg,
                                                std::string const& modulelabel,
                                                STRINGLIKEs... names)
{
  // Make sure all the arguments are convertible to std::string.
  // The parameter names are not used directly but are kept for consistency
  // and to easily tell the order of the grid points (if they match the
  // column order in the grid_file)
  static_assert(
    std::conjunction_v<std::is_convertible<STRINGLIKEs, std::string>...>,
    "\n\nCosmoSIS error!\nAll trailing arguments in "
    "load_grid_points_wall_of_numbers must be convertible to string.\n\n");
  constexpr std::size_t n_axes = sizeof...(STRINGLIKEs);
  std::array<std::string, n_axes> const axis_names{
    std::forward<STRINGLIKEs>(names)...};
  std::array<std::vector<double>, n_axes> axes;
  detail::get_grid_axes_from_file(cfg, modulelabel, axes);
  return detail::make_grid_wall_of_numbers(axes);
}

#endif
