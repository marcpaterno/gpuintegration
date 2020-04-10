#ifndef Y3_CLUSTER_CPP_META_HH
#define Y3_CLUSTER_CPP_META_HH

#include <vector>

namespace y3_cluster {
  namespace detail {

    // cartesian_product(f, v...) means "do `f` for each element of
    // cartesian product of v..."

    // Base case: execute the now-fully-bound sub-accumulator, to execute
    // the originally-bound 'f' using the bound set of arguments.
    template <typename F>
    void
    cartesian_product(F f)
    {
      f();
    }

    // When we have a head and tail, loop over each element in the head,
    // binding each into a nested sub-accumulator, and run that sub-accumulator
    // over the remaining input arguments.
    template <typename F, typename H, typename... Ts>
    void
    cartesian_product(F f,
                      std::vector<H> const& head,
                      std::vector<Ts> const&... tail)
    {
      for (H const& h : head) {
        auto sub_accumulator = [&h, &f](Ts const&... ts) { f(h, ts...); };
        cartesian_product(sub_accumulator, tail...);
      }
    }
  }
}

#endif
