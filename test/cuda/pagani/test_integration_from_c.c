#include "test/cuda/pagani/do_integration_from_c.h"
#include <stdio.h>
#include <math.h>

int
main()
{
  double result = 0.0;
  int const rc = do_integration_from_c(&result);
  printf("Status: %i   result: %f\n", rc, result);
  if (rc != 0) return rc;
  // The relative tolerance for the integration is hardwired to 1.0e-6
  // in do_integration_from_c.cu.
  //
  // The correct answer for this integral is exactly 1/4.
  double achieved_relative_error = fabs(result - 0.25) / 0.25;
  if (achieved_relative_error > 1.0e-6) {
    printf("Failed to achieve relative error of 1.0e-6\n");
    printf("achieved relative error of %g\n", achieved_relative_error);
    return 1;
  }
  return 0;
}
