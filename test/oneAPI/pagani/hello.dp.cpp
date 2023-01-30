#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include <CL/sycl.hpp>
// #include <dpct/dpct.hpp>
#include "catch2/catch.hpp"
#include <stdio.h>

void
cuda_hello(const sycl::stream& stream_ct1)
{
  stream_ct1 << "Hello World from GPU!\n";
}

int
main()
{
  auto q_ct1 = sycl::queue(sycl::gpu_selector());
  q_ct1.submit([&](sycl::handler& cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(
      sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { cuda_hello(stream_ct1); });
  });
  dpct::get_current_device().queues_wait_and_throw();
  return 0;
}
