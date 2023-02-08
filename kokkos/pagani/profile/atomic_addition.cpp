#include <iostream>
#include <array>
#include <numeric>
#include <vector>

#include <Kokkos_Core.hpp>
#include "common/kokkos/cudaMemoryUtil.h"
void
atomic_addition(ViewVectorDouble src, ViewVectorDouble out, size_t size)
{
  uint32_t nBlocks = size;
  uint32_t nThreads = 64;

  Kokkos::TeamPolicy<Kokkos::LaunchBounds<64, 18>> mainKernelPolicy(nBlocks,
                                                                    nThreads);
  Kokkos::parallel_for(
    "Phase1", mainKernelPolicy, KOKKOS_LAMBDA(const member_type team_member) {
      size_t tid = team_member.league_rank() * team_member.team_size() +
                   team_member.team_rank();
      size_t total_num_threads = nBlocks * nThreads;

      for (size_t i = tid; i < size; i += total_num_threads) {
        for (int i = 0; i < 8; ++i)
          Kokkos::atomic_add(&out[team_member.team_rank()], src[i]);
      }
    });
}

int
main()
{
  Kokkos::initialize();
  const size_t num_threads = 64;

  std::vector<double> src;
  src.resize(32768 * 1025 * 2);
  std::iota(src.begin(), src.end(), 1.);

  std::array<double, num_threads> output = {0.};

  std::cout << "size:" << src.size() << std::endl;
  std::cout << "Memory:" << src.size() * 8 / 1e9 << "GB\n";

  ViewVectorDouble d_src("d_src", src.size());
  ViewVectorDouble d_output("d_output", output.size());

  auto h_src = Kokkos::create_mirror_view(d_src);
  auto h_output = Kokkos::create_mirror_view(d_output);

  for (size_t i = 0; i < src.size(); ++i)
    h_src[i] = src[i];
  Kokkos::deep_copy(d_src, h_src);

  size_t num_blocks = src.size() / num_threads;

  atomic_addition(d_src, d_output, src.size());
  Kokkos::deep_copy(h_output, d_output);

  for (size_t i = 0; i < h_output.size(); ++i)
    printf("output %i, %e\n", i, h_output[i]);
  Kokkos::finalize();
  return 0;
}
