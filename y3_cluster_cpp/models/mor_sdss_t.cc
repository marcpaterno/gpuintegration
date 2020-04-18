#include "mor_sdss_t.hh"
#include "utils/interp_2d.hh"

namespace y3_cluster {

  Interp2D const MOR_sdss::sig_interp =
    Interp2D(test_sigintr, test_lsat, sig_skewnorml_flat);

  Interp2D const MOR_sdss::skews_interp =
    Interp2D(test_sigintr, test_lsat, skews);
}