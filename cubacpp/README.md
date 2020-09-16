## Multidimensional Integration in C++

`cubacpp` provides a C++ binding for the excellent [CUBA](http://www.feynarts.de/cuba)
library, and to a lesser extent, to the GSL integration library [GSL](https://www.gnu.org/software/gsl/manual/html_node/Numerical-Integration.html).
`cubacpp` provides no new integration facilities. It merely provides a
more convenient syntax for use of the CUBA (and GSL) integration routines, made possible
by the features of the C++ programming language.

## Obtaining and building the software

### Obtaining cubacpp

Please clone the repository! If you wish to contribute, fork the repository, and
send me pull requests.

### Obtaining  and building CUBA

To use `cubacpp`, you must have a copy of the CUBA software, available from
<http://www.feynarts.de/cuba>. The directory `scripts` (in this repository) contains a script for
downloading and building CUBA into a dynamic library on a Linux system. This is
the way CUBA is used in the projects in which I participate. You may need to
adjust the variable settings in the script. Building CUBA requires only a
C compiler, compliant with the C89 standard.

Under macOS, you can install CUBA using Homebrew:
```bash
$ brew install cuba
```
While this installs a static library version of CUBA,
on macOS this library is compatible with use in dynamic libraries.

### Obtaining and building GSL

The GSL interface has been tested with version 2.4, and will likely work with
higher (and perhaps some lower) versions. It can be found
[here](https://www.gnu.org/software/gsl/).

On Ubuntu derivatives, GSL is available through the standard package manager:

```bash
$ sudo apt-get install libgsl-dev
```

On Fedora, GSL is available through the standard
package manager:

```bash
$ sudo dnf install gsl-devel
```

On older Linux distributions, is may be necessary to download and build GSL yourself.
Building GSL requires only a C compiler, compliant with the C89 standard.

### Building cubacpp

There is nothing you need to build for `cubacpp` itself; the library is header-only. To
use `cubacpp`, you will need a compiler that support C++17. GCC v7 meets the
need, when run with the flag `-std=c++17`. Clang v5 meets the need, also when
run with flag `-std=c++17`. Apple Clang as of version 9.2, when run with `-std=c++1z`,
also meets the need. I would be happy to receive reports of other 
compilers that also suffice.

The `test` directory contains an example showing how to use the integration
functions provided by `cubacpp`. It also contains a CMakeLists.txt file to
be used by CMake, to create a build. You may need to direct CMake to the
compiler you wish to use. Most likely, you will also have to direct CMake
to the location of your CUBA installation.
For example, on a machine with GCC 7.x installed
as `g++-7` (which can be obtained through Homebrew on macOS), and with
CUBA installed under $HOME, the following command would create a `Makefile`
to build the testing programs.

```bash
$ cmake -DCUBA_DIR=$HOME -DCMAKE_CXX_COMPILER=$(which g++-7) -DCMAKE_BUILD_TYPE=Release  <path-to-CMakeLists.txt-file>
```

## Using cubacpp

The user interface of `cubacpp` consists of a few function templates,
and a few structs,
one of each for
each of the wrapped integration routines.
Note that not all of the routines supplied by CUBA and GSL are wrapped;
pull requests will be welcome!

The simplest use of `cubacpp` is through the function templates.
Each of the function
templates takes, as its first argument, the function to be integrated. This
integrand can be supplied as any C++ callable object:

1. A free function (e.g., a C-style function).
2. An object of a class that supplies a function call operator (`operator()`),
   including an instance of `std::function`.
3. A lambda expression.
4. For the CUBA integration routines, a volume of integration.

In each case, the supplied callable object must be callable with one or more
arguments of type `double` (`Cuhre` has the restriction that the function
must have at least 2 function call arguments); thus the function arguments must either be `double`,
or some other numeric type convertible to `double`. CUBA can calculate vector-valued
integrals; to take advantage of this facility, the callable object is allowed to
return either (one) `double`, or an `std::array<double, N>`, for any positive integral
value of `N`, or an `std::vector<double>`. In this last case, it is critical that
the function being integrated must return a vector of the same length on every
call. This can not be verified by the compiler; it is up to the user to ensure
that this is the case.

Since GSL integrators operate on functions of one dimension,
and can not handle vector-valued functions,
the GSL wrappers will only work for callable objects which take
a single double, and return a double (not `std::array` or `std::vector`).

### Using the CUBA function templates

The detailed description of the CUBA integration routines are available
at the [CUBA web site](http://www.feynarts.de/cuba). We describe here how the arguments of
the CUBA functions are determined from the `cubacpp` call.

The CUBA algorithms integrate the supplied integrand over the unit hypercube.
The CUBA manual notes that a simple transformation of variables
allows for other volumes of integration.
In order to make the handling of other volumes of integration easier,
each of the `cubacpp` wrappers for CUBA routines accepts an integration volume,
which defaults to the unit hypercube.
The `cubacpp` integration functions handle the transformation of variables automatically.


#### Common CUBA arguments

These arguments are common to all the CUBA integration routines. Note that
the `ll` versions (those using `long long int` arguments) 
 of the CUBA routines are used by `cubacpp`, so several
of the arguments that are listed in the CUBA documentation as being of
type `int` are in fact of type `long long int`, as described in section 7.4
of the CUBA documentation.

1. `int ndim`: This value is determined automatically from the
    number of arguments required to call the user-supplied
    integrand. Note that the Cuhre algorithm requires an integrand
    which takes at least two arguments; trying to use it with
    a function of fewer arguments will produce a compilation
    failure in `cubacpp`, rather then the runtime error of
    CUBA.
2. `int ncomp`: The value is determined automatically from the
   number of `double` values returned by a call to the user-supplied
   integrand. 
3. `integrand`: This function is automatically generated based
   on the user-supplied integrand, supplied as any callable C++
   object (e.g. a free function, an instance of a class with
   a function call operator, or a lambda expression).
4. `void* userdata`: This value passed is a pointer to the
   function or other callable object passed as the integrand.
5. `int nvec`: The value is set to 1 by `cubacpp`.
6. `double epsrel, epsabs`: The values provided to the
   `epsrel` and `epsabs` arguments of the `cubacpp` function
   are passed to CUBA.
7. `int flags`: The value provided by the `flags` argument
   of the `cubacpp` functions is passed to CUBA.
8. `int seed`: The value is set to 0 by `cubacpp`.
9. `long long mineval`: The value supplied by the
   user is passed to CUBA. A default of 0 is provided
   by `cubacpp`.
10. `long long int maxeval`: The value supplied by the user
    is passed to CUBA. A default of 50,000 is provided
    by `cubacpp`.
11. `const char* statefile`: The value `nullptr` is supplied
    to CUBA. Using a stored state file seems problematic in an
    MPI environment.
12. `void* spin`: The value `nullptr` is passed to CUBA. Using
    CUBA to manage multiple operating system processes is
    problematic in an MPI environment.
13. `long long int neval`: The value returned by CUBA in this
    output argument is returned in the `integration_result`
    struct returned by all the `cubacpp` integration functions,
    in the data member `neval`.
14. `int fail`: The value returned by CUBA in this output
    argument is returned in the `integration_result` struct
    returned by all the `cubacpp` integration functions, in
    the data member `status`.
15. `double integral`: The value returned by CUBA in this output
    argument is returned in the `integration_result` struct
    returned by all the `cubacpp` integration functions, in the
    data member `value`.
16. `double error`: The value returned by CUBA in this output
    argument is returned in the `integration_result` struct
    returned by all the `cubacpp` integration functions, in the
    data member `error`.
17. `double prob`: The value returned by CUBA in this output
    argument is returned in the `integration_result` struct
    returned by all the `cubacpp` integration functions, in the
    data member `prob`.

#### Vegas-specific arguments

1. `long long int nstart`: The value supplied by the user is passed
   to Vegas. A default of 1000 is supplied by `cubacpp`.
2. `long long int nincrease`: The value supplied by the user is passed
   to Vegas. A default of 500 is supplied by `cubacpp`.
3. `long long int nbatch`: The value supplied by the user is passed
   to Vegas. A default of 1000 is supplied by `cubacpp`.
4. `integer gridno`: The value 0 passed to Vegas.

#### Suave-specific arguments

1. `int* nregions`: The value return by CUBA in this output argument
   is return in the `integration_result` struct returned
   by the all the `cubacpp` integration functions, in the 
   data member `nregions`. 
   interface.
2. `long long int nnew`: The value supplied by the user is passed
   to Suave. A default of 1000 is supplied by `cubacpp`.
3. `long long int nmin`: The value supplied by the user is passed
   to Suave. A default of 2 is provided by `cubacpp`.
4. `double flatness`: The value supplied by the user is passed to
   Suave. A default of 25.0 is provided by `cubacpp`.

#### Cuhre-specific arguments

1. `int* nregions`: The value return by CUBA in this output argument
   is return in the `integration_result` struct returned
   by the all the `cubacpp` integration functions, in the 
   data member `nregions`. 
   interface.
2. `int key`: The value supplied by the user is passed to Cuhre.
   A default value of -1 is supplied by `cubacpp`. If -1 is passed,
   `cubacpp` sets the degree of the rule to the maximum allowed,
   based on the dimension of the integrand.

#### Divonne-specific arguments

An interface to the Divonne algorithm is not yet implemented.
A pull request would be welcome!

### Using the GSL function templates

Currently, `cubacpp` only wraps four GSL integrators -
[QNG](https://www.gnu.org/software/gsl/doc/html/integration.html#c.gsl_integration_qng),
[QAG](https://www.gnu.org/software/gsl/doc/html/integration.html#qag-adaptive-integration),
[CQUAD](https://www.gnu.org/software/gsl/doc/html/integration.html#cquad-doubly-adaptive-integration), and
[QAWF](https://www.gnu.org/software/gsl/doc/html/integration.html#qawf-adaptive-integration-for-fourier-integrals).

The first three are general-purpose one dimensional integrators.
The fourth, QAWF, is described futher below.

QAG and CQUAD are adaptive,
while QNG is non-adaptive. Since these GSL algorithms integrate an explicit
integration range, rather than assuming the unit hypercube, these integrators
all take `range_start` and `range_end` constructor arguments. They default to
0 and 1, respectively, for consistency with CUBA.

For instance,

```cpp
auto square = [](double x) { return x * x; };
cubacpp::QAG qag(-10.0, 3.0);
auto result = qag.integrate(square, 1e-5, 1e-18);
```

Will integrate the function `x^2` on the interval [-10, 3]. CQUAD may be
initialized the same way.

Since it may be desirable to use the same integrator for functions of different
ranges, both QAG and CQUAD structs have a `with_range()` member function to change
the range after-the-fact:

```cpp
auto res1 = qag.with_range(0, 12)
               .integrate(square, 1e-5, 1e-18);
auto res2 = qag.with_range(12, 40)
               .integrate(square, 1e-5, 1e-18);
```

`res1` will hold the value of the integration from 0 to 12, and `res2` the
value of the integration from 12 to 40.

#### Common GSL Arguments for QNG, QAG and CQUAD

1. `double range_start`: The beginning of the integration range. Discussed
   above.
2. `double range_end`: The end of the integration range. Discussed above.

#### QAG-specific arguments

1. `int key`: QAG can use a number of different Gauss-Konrod rules, of 15, 21,
   31, 41, 51, or 61 points. GSL provides symbolic macros for this parameter,
   `GSL_INTEG_GAUSS15`, `GSL_INTEG_GAUSS21`, etc. Defaults to the 61-point
   rule.
2. `size_t limit`: The maximum number of intervals to allocate space for.
   Defaults to 10, maximum of 20. Consult the GSL docs [here](https://www.gnu.org/software/gsl/doc/html/integration.html#c.gsl_integration_workspace_alloc).

#### CQUAD-specific arguments

1. `size_t n`: The maximum number of intervals to allocate space for.
   Defaults to 100. Consult the GSL docs [here](https://www.gnu.org/software/gsl/doc/html/integration.html#c.gsl_integration_cquad_workspace_alloc).

#### The QAWF integration routine

The QAWF integration routine implements adaptive integration for Fourier integrals.
It integrates over a semi-infinite range, so only the lower bound of integration `range_start` can be specified.
See the GSL web page linked above for details.

### Using the integration structs

Because there are so many function parameters that may be passed to each of the
function templates, `cubacpp` also contains a struct for each integration
algorithm, each of which contains a data member for each parameter that controls
the integral evaluation, except for `epsrel` and `epsabs`. The individual
parameters may be set by assigning to the relevant data members. The integration
may then be done by calling the `integrate` member function of the struct,
passing the function to be integrated, and the values of `epsrel` and `epsabs`.

## Issues with parallelism

Because the main use case for `cubacpp` is in an MPI environment, use of CUBA's
facilities for forking multiple processes to speed calculations is problematic.
I recommend that users call `cubacores(0, 0)` before executing any `cubacpp`
integration.
 
## Special notes for users of CosmoSIS

The `scripts` directory contains a `Dockerfile` suitable for use with the
_comsosis-docker_ delivery of CosmoSIS. Please make sure your version of
_cosmosis-docker_ is sufficiently recent; this means the version of the
GCC compiler is v7 or newer.
