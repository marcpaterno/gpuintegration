#include <CL/sycl.hpp>
#include <iostream>
#include "oneAPI/pagani/demos/new_time_and_call.dp.hpp"
#include "common/oneAPI/integrands.hpp"

int
main(int argc, char** argv)
{
  int num_repeats = argc > 1 ? std::stoi(argv[1]) : 11;
    SinSum_3D sinsum_3D;
    SinSum_4D sinsum_4D;
    SinSum_5D sinsum_5D;
    SinSum_6D sinsum_6D;
    SinSum_7D sinsum_7D;
    SinSum_8D sinsum_8D;

	quad::Volume<double, 3> vol3;
	quad::Volume<double, 4> vol4;
	quad::Volume<double, 5> vol5;
	quad::Volume<double, 6> vol6;
	quad::Volume<double, 7> vol7;
	quad::Volume<double, 8> vol8;

	call_cubature_rules<SinSum_3D, 3>(sinsum_3D, vol3, num_repeats);
	call_cubature_rules<SinSum_4D, 4>(sinsum_4D, vol4, num_repeats);
	call_cubature_rules<SinSum_5D, 5>(sinsum_5D, vol5, num_repeats);
	call_cubature_rules<SinSum_6D, 6>(sinsum_6D, vol6, num_repeats);
	call_cubature_rules<SinSum_7D, 7>(sinsum_7D, vol7, num_repeats);
	call_cubature_rules<SinSum_8D, 8>(sinsum_8D, vol8, num_repeats);
    return 0;
}

