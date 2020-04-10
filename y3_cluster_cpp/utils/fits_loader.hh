#ifndef Y3CLUSTER_FITS_LOADER_HH
#define Y3CLUSTER_FITS_LOADER_HH

#include "interp_1d.hh"

#include <exception>
#include <fitsio.h>
#include <memory>
#include <string>

namespace y3_cluster {
  namespace {
    // Loads a single column (named `colname`) of doubles from a fits
    // table. If it works, returns {true, column}, if it fails, returns
    // {false, <empty vector>}.
    std::pair<bool, std::vector<double>>
    load_fits_column(fitsfile* file,
                     const char* colname_,
                     long nrows,
                     int* status)
    {
      if (nrows < 0)
        return {false, {}};

      std::string colname = colname_;
      int colno = 0;
      fits_get_colnum(file, CASEINSEN, &colname[0], &colno, status);

      std::vector<double> output(nrows);
      fits_read_col(file,
                    TDOUBLE,    // datatype
                    colno,      // colno
                    1,          // firstrow
                    1,          // firstelem
                    nrows,      // nelements
                    NULL,       // nulval
                    &output[0], // array
                    NULL,       // anynul
                    status);

      if (*status)
        return {false, {}};
      return {true, output};
    }
  } // namespace

  /* Returns a pair of {zgrid, {pz1, pz2, pz3, pz4}} */
  std::pair<std::vector<double>, std::vector<std::vector<double>>>
  load_pzsource_data(const std::string& filename)
  {
    fitsfile* file = NULL;
    int status = 0;

    // Why do these functions return an `int` if the return value is not
    // supposed to be used? Am I missing something? It must mean
    // something
    fits_open_file(&file, filename.c_str(), READONLY, &status);

    // Find the table - `nz_source`
    char tbl_name[] = "nz_source";
    fits_movnam_hdu(file, BINARY_TBL, tbl_name, 1, &status);

    // How many rows do we need?
    long nrows = 0;
    fits_get_num_rows(file, &nrows, &status);

    if (nrows > 0) {
      // First: read redshifts - Z_MID
      auto [success1, zs] = load_fits_column(file, "Z_MID", nrows, &status);
      bool success = success1;

      if (success) {
        // If that worked - read in each bin one at a time
        std::vector<std::string> colnames{"BIN1", "BIN2", "BIN3", "BIN4"};
        std::vector<std::vector<double>> pzs;
        for (auto& colname : colnames) {
          auto [success2, bin] =
            load_fits_column(file, colname.c_str(), nrows, &status);
          success = success && success2;

          if (!success)
            break;

          // Record this PZSOURCE distribution
          pzs.push_back(bin);
        }

        // If it worked - then great!
        if (success)
          return {zs, pzs};
      }
    }

    fits_close_file(file, &status);

    // If anything failed - throw an error
    // NB: This can be done just at the end, b/c cfitsio routines are each
    // passed the status of the previous routine, and do nothing if any
    // previous call failed. Neat design.
    if (status) {
      char errmsg[FLEN_ERRMSG];
      fits_get_errstatus(status, errmsg);
      errmsg[FLEN_ERRMSG - 1] = '\0';
      throw std::runtime_error(std::string("Fits error: ") + errmsg);
    } else
      throw std::runtime_error(
        "Could not find one of (Z_MID, BIN1, BIN2, BIN3, BIN4) in fits file");
  }

  std::vector<std::shared_ptr<Interp1D const>>
  load_pzsources(const std::string& filename)
  {
    const auto [zgrid, pzs] = load_pzsource_data(filename);
    std::vector<std::shared_ptr<Interp1D const>> output;
    for (auto pz : pzs)
      output.push_back(std::make_shared<Interp1D const>(zgrid, pz));
    return output;
  }
} // namespace y3_cluster

#endif
