#ifndef Y3_CLUSTER_CPP_MAKE_INTEGRATION_VOLUME_HH
#define Y3_CLUSTER_CPP_MAKE_INTEGRATION_VOLUME_HH

#include "cubacpp/array.hh"
#include "cubacpp/integration_volume.hh"
#include "utils/datablock.hh"
#include "utils/datablock_reader.hh"
#include "utils/meta.hh"

#include <array>
#include <string>
#include <type_traits>
#include <vector>

namespace y3_cluster {
  // These variadic function templates takes as arguments:
  //   1. a cosmosis::DataBlock (by reference),
  //   2. the name of the module being configured, and
  //   3. one or more arguments that are convertible to strings.
  //
  // Assuming your class's name is ClsName, it should be used like:
  //    DataBlock cfg;
  //    auto vols = make_integration_volumes(cfg, ClsName::module_label(), "x",
  //    "y", "z");
  //
  // The function returns a vector of IntegrationVolume<N>,
  // where N is the number of names provided.
  // This version expects the configuration to have come from the
  // "wall of numbers" style of configuration.`
  template <typename... Ts>
  std::vector<cubacpp::IntegrationVolume<sizeof...(Ts)>>
  make_integration_volumes_wall_of_numbers(cosmosis::DataBlock& cfg,
                                           std::string const& modulelabel,
                                           Ts... names);

  template <typename... Ts>
  std::vector<cubacpp::IntegrationVolume<sizeof...(Ts)>>
  make_integration_volumes_cartesian_product(cosmosis::DataBlock& cfg,
                                             std::string const& modulelabel,
                                             Ts... names);

  namespace detail {
    template <std::size_t N>
    using integration_boundary = cubacpp::array<N>;
    template <typename T>
    using pair_of_vector = std::pair<std::vector<T>, std::vector<T>>;

    // We have to use int N, rather than std::size_t N, because that is what the
    // class template cubacpp::array<N> expects; this in turn is determined by
    // Eigen.
    template <std::size_t N>
    pair_of_vector<integration_boundary<N>>
    get_integration_boundaries_wall_of_numbers(
      cosmosis::DataBlock& cfg,
      std::string const& modulelabel,
      std::array<std::string, N> const& names)
    {
      if (names.empty())
        return {};

      // The first parameter gets special handling, because we use it to
      // determine how many integration volumes we shall produce.
      auto lows = get_vector_double(cfg, modulelabel, names[0] + "_low");
      std::size_t const nvolumes = lows.size();

      auto highs = get_vector_double(cfg, modulelabel, names[0] + "_high");
      if (nvolumes != highs.size()) {
        // TODO: Improve this error handling.
        throw std::runtime_error(names[0] + " bad, bad user!");
      }

      std::vector<integration_boundary<N>> lowbounds(nvolumes);
      std::vector<integration_boundary<N>> highbounds(nvolumes);
      auto fill_bounds = [&lowbounds, &lows, &highbounds, &highs](
                           std::size_t iname, std::size_t ivol) {
        lowbounds[ivol][iname] = lows[ivol];
        highbounds[ivol][iname] = highs[ivol];
      };

      for (std::size_t ivol = 0; ivol != nvolumes; ++ivol)
        fill_bounds(0, ivol);

      // All other parameters are handled identically to each other. Each must
      // have the same number of integration volumes.
      for (std::size_t iname = 1; iname != N; ++iname) {
        lows = get_vector_double(cfg, modulelabel, names[iname] + "_low");
        highs = get_vector_double(cfg, modulelabel, names[iname] + "_high");
        if (nvolumes != lows.size() || nvolumes != highs.size()) {
          // TODO: Improve this error handling.
          throw std::runtime_error(names[iname] + " bad, bad user!");
        }
        for (std::size_t ivol = 0; ivol != nvolumes; ++ivol)
          fill_bounds(iname, ivol);
      }
      return {lowbounds, highbounds};
    }

    template <typename... Ts>
    std::vector<std::array<double, sizeof...(Ts)>>
    make_cartesian_product_splatted(std::vector<Ts> const&... bounds)
    {
      // Make sure the vectors are carrying floating point numbers;
      // we convert everything to double, 'cause doing otherwise is hard.
      static_assert(std::conjunction_v<std::is_floating_point<Ts>...>);

      std::vector<std::array<double, sizeof...(Ts)>> res;
      auto accumulator = [&res](Ts const&... ts) {
        // Construct an array from all the elements we pass in ts...
        res.push_back({ts...});
      };
      detail::cartesian_product(accumulator, bounds...);
      return res;
    }

    template <std::size_t... Is>
    std::vector<cubacpp::array<sizeof...(Is)>>
    make_boundaries_cartesian_product_aux(
      std::array<std::vector<double>, sizeof...(Is)> const& boundaries,
      std::index_sequence<Is...> /* unused */)
    {
      // The cartesian product facilities work in terms of std::array, while
      // integration_boundary uses cubacpp::array, which in turn uses
      // Eigen::array. We have to do the translation here.
      std::vector<std::array<double, sizeof...(Is)>> bounds =
        make_cartesian_product_splatted(boundaries[Is]...);
      std::vector<cubacpp::array<sizeof...(Is)>> result;
      result.reserve(bounds.size());
      for (auto const& current_bound : bounds) {
        cubacpp::array<sizeof...(Is)> tmp;
        for (std::size_t i = 0; i != sizeof...(Is); ++i)
          tmp[i] = current_bound[i];
        result.push_back(tmp);
      }
      return result;
    }

    template <std::size_t N>
    std::vector<integration_boundary<N>>
    make_boundaries_cartesian_product(
      std::array<std::vector<double>, N> const& boundaries)
    {
      return detail::make_boundaries_cartesian_product_aux(
        boundaries, std::make_index_sequence<N>());
    }

    template <std::size_t N>
    pair_of_vector<integration_boundary<N>>
    get_integration_boundaries_cartesian_product(
      cosmosis::DataBlock& cfg,
      std::string const& modulelabel,
      std::array<std::string, N> const& names)
    {
      if (names.empty())
        return {};

      using vec = std::vector<double>;
      std::array<vec, N> low_boundaries;
      std::array<vec, N> high_boundaries;

      for (std::size_t i = 0; i != N; ++i) {
        low_boundaries[i] =
          get_vector_double(cfg, modulelabel, names[i] + "_low");
        high_boundaries[i] =
          get_vector_double(cfg, modulelabel, names[i] + "_high");
      }

      std::vector<integration_boundary<N>> lows =
        detail::make_boundaries_cartesian_product(low_boundaries);
      std::vector<integration_boundary<N>> highs =
        detail::make_boundaries_cartesian_product(high_boundaries);
      return {lows, highs};
    }

  } // namespace detail
} // namespace y3_cluster

template <typename... Ts>
std::vector<cubacpp::IntegrationVolume<sizeof...(Ts)>>
y3_cluster::make_integration_volumes_wall_of_numbers(
  cosmosis::DataBlock& cfg,
  std::string const& modulelabel,
  Ts... names)
{
  // Make sure that all arguments are convertible to std::string.
  static_assert(std::conjunction_v<std::is_convertible<Ts, std::string>...>,
                "\n\nCosmoSIS error!\nAll trailing arguments in "
                "make_integration_volumes_wall_of_numbers must be convertible "
                "to string.\n\n");

  constexpr std::size_t n = sizeof...(Ts);
  std::array<std::string, n> stringnames{std::forward<Ts>(names)...};

  auto [lows, highs] = detail::get_integration_boundaries_wall_of_numbers(
    cfg, modulelabel, stringnames);

  std::vector<cubacpp::IntegrationVolume<n>> result;
  result.reserve(lows.size());
  for (std::size_t i = 0; i != lows.size(); ++i) {
    result.emplace_back(lows[i], highs[i]);
  }

  return result;
}

template <typename... Ts>
std::vector<cubacpp::IntegrationVolume<sizeof...(Ts)>>
y3_cluster::make_integration_volumes_cartesian_product(
  cosmosis::DataBlock& cfg,
  std::string const& modulelabel,
  Ts... names)
{
  // Make sure that all arguments are convertible to std::string.
  static_assert(std::conjunction_v<std::is_convertible<Ts, std::string>...>,
                "\n\nCosmoSIS error!\nAll trailing arguments in "
                "make_integration_volumes_cartesian_product must be "
                "convertible to string.\n\n");

  constexpr std::size_t n = sizeof...(Ts);
  std::array<std::string, n> stringnames{std::forward<Ts>(names)...};

  auto [lows, highs] = detail::get_integration_boundaries_cartesian_product(
    cfg, modulelabel, stringnames);

  std::vector<cubacpp::IntegrationVolume<n>> result;
  result.reserve(lows.size());
  for (std::size_t i = 0; i != lows.size(); ++i) {
    result.emplace_back(lows[i], highs[i]);
  }

  return result;
}

#endif
