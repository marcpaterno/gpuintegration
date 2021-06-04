int do_integration_from_c(double*);

#include <stdio.h>

int main() {
  double result;
  result = 0.0;
  int rc ;
  rc = do_integration_from_c(&result);
  printf("Status: %i   result: %f\n", rc, result);
  return rc;
}
