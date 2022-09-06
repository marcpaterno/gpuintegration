#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

void
cuda_hello(const sycl::stream &stream_ct1)
{
  stream_ct1 << "Hello World from GPU!\n";
}

int
main()
{
  printf("Before calling the kernel\n");
  dpct::get_default_queue().submit([&](sycl::handler& cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                     [=](sycl::nd_item<3> item_ct1) {
                       cuda_hello(stream_ct1);
                     });
  });
  dpct::get_current_device().queues_wait_and_throw();
  printf("After calling the kernel\n");
  return 0;
}
