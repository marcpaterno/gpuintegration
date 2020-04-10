#include "lc_lt_t.hh"
using namespace y3_cluster;

namespace {

  // Define zt_max and l_max large enough to be safe.
  double constexpr zt_max = 2.0;
  double constexpr lt_max = 300.0;

  std::array<double, 6> constexpr zt_bins = {0.1, 0.15, 0.2, 0.25, 0.3, zt_max};

  std::array<double, 23> constexpr lt_bins = {
    1.,          3.,          5.,          7.,          9.,         12.,
    15.55555534, 20.,         24.,         26.11111069, 30.,        36.66666412,
    40.,         47.22222137, 57.77777863, 68.33332825, 78.8888855, 89.44444275,
    100.,        120.,        140.,        160.,        lt_max};

  std::array<double, 110> constexpr sigma_arr = {
    8.12748540e-01, 1.32292795e+00, 1.70726638e+00, 1.74780086e+00,
    2.01806368e+00, 2.23657343e+00, 2.40010333e+00, 2.70710073e+00,
    3.12735013e+00, 3.21122400e+00, 3.39146912e+00, 3.74153501e+00,
    4.28415873e+00, 4.68862761e+00, 5.30421671e+00, 6.25441698e+00,
    8.11481918e+00, 1.00779859e+01, 1.25872284e+01, 1.57642842e+01,
    1.87555074e+01, 1.85680995e+01, 8.97538569e-01, 1.45653675e+00,
    1.83089463e+00, 2.03780350e+00, 2.36165900e+00, 2.26214818e+00,
    2.63760517e+00, 2.74378351e+00, 3.04664020e+00, 3.27064033e+00,
    3.72379653e+00, 4.11115833e+00, 4.25761957e+00, 5.18025378e+00,
    5.97535357e+00, 6.39831098e+00, 7.24710232e+00, 7.61317257e+00,
    1.05181754e+01, 1.50297905e+01, 2.14436250e+01, 2.31502787e+01,
    1.03442952e+00, 1.57201455e+00, 2.03178973e+00, 2.04579518e+00,
    2.49762679e+00, 2.64317300e+00, 2.95512309e+00, 3.59622507e+00,
    3.71391424e+00, 3.63535427e+00, 3.97900209e+00, 4.76320322e+00,
    5.18783911e+00, 5.31311098e+00, 6.31225872e+00, 6.98035094e+00,
    9.03881302e+00, 1.02189303e+01, 1.22805981e+01, 1.59530638e+01,
    2.17144827e+01, 2.88266797e+01, 1.13029177e+00, 1.72414335e+00,
    2.16423058e+00, 2.49632582e+00, 2.82255646e+00, 3.00348102e+00,
    3.44326967e+00, 3.80849905e+00, 4.38262038e+00, 4.49771781e+00,
    4.83905199e+00, 5.57251073e+00, 5.85608999e+00, 6.59513668e+00,
    7.33061542e+00, 9.00693434e+00, 9.62863263e+00, 9.89458179e+00,
    9.78875159e+00, 1.00866611e+01, 1.10241365e+01, 1.46325049e+01,
    1.04521441e+00, 1.70304884e+00, 2.17899895e+00, 2.43057443e+00,
    2.96320069e+00, 3.33061104e+00, 3.60372525e+00, 4.09736301e+00,
    4.40883299e+00, 4.74553817e+00, 5.17065665e+00, 5.76663140e+00,
    6.16994764e+00, 7.02731129e+00, 7.68272673e+00, 8.76098853e+00,
    9.89865696e+00, 1.04363822e+01, 1.22443047e+01, 1.56979618e+01,
    2.19841273e+01, 2.53251473e+01};

  std::array<double, 110> constexpr fmsk_arr = {
    3.27308389e-01, 3.12345575e-01, 3.00618261e-01, 2.57318771e-01,
    2.39648376e-01, 2.05210932e-01, 1.47652380e-01, 1.25691770e-01,
    1.06213660e-01, 9.42416400e-02, 8.31716990e-02, 6.29795509e-02,
    5.60964689e-02, 3.26932365e-02, 1.82612679e-02, 1.07365552e-02,
    4.47057862e-03, 2.21375725e-03, 1.39615480e-03, 2.74752696e-04,
    1.00957318e-05, 9.48807606e-06, 3.01774037e-01, 2.71888061e-01,
    3.24287420e-01, 3.03626578e-01, 2.37934127e-01, 2.06487010e-01,
    1.61807110e-01, 1.34451663e-01, 1.12758878e-01, 9.68591679e-02,
    8.31834698e-02, 5.85830468e-02, 5.16183659e-02, 3.55628244e-02,
    3.33889847e-02, 2.17171395e-02, 2.08578148e-02, 1.15521407e-02,
    8.67821664e-03, 2.88167212e-03, 1.39888087e-03, 3.24969353e-04,
    2.80923414e-01, 4.16431270e-01, 3.21967193e-01, 3.56490356e-01,
    2.80653628e-01, 2.01611248e-01, 1.76399513e-01, 1.23683685e-01,
    1.08532076e-01, 7.91462585e-02, 6.93724355e-02, 4.70948662e-02,
    4.79773955e-02, 4.24421873e-02, 3.77386603e-02, 3.17676935e-02,
    3.17779563e-02, 1.64718442e-02, 1.65275011e-02, 4.42346465e-03,
    8.12635405e-04, 7.19134727e-04, 2.95490092e-01, 5.08366741e-01,
    3.61041273e-01, 3.88828847e-01, 3.06498186e-01, 2.53756806e-01,
    1.84161754e-01, 1.44432550e-01, 1.04835343e-01, 9.97635940e-02,
    9.33874739e-02, 5.69355021e-02, 4.58331855e-02, 3.17481163e-02,
    2.08137614e-02, 1.31490621e-02, 9.62636147e-03, 1.42006662e-02,
    1.37248001e-02, 2.12084787e-02, 2.44451133e-02, 1.37853248e-02,
    4.28801627e-01, 4.54542456e-01, 3.63503306e-01, 3.67703291e-01,
    3.24505168e-01, 2.60670545e-01, 2.22207057e-01, 1.49103193e-01,
    1.02423045e-01, 9.58746427e-02, 7.59982205e-02, 5.69084212e-02,
    5.13631282e-02, 4.41356887e-02, 3.26416460e-02, 3.24171668e-02,
    2.49491123e-02, 1.76920787e-02, 1.17437463e-02, 5.64348319e-03,
    1.17441933e-05, 7.27777308e-04};

  std::array<double, 110> constexpr fprj_arr = {
    3.13611172e-03, 3.15339208e-03, 3.14040215e-03, 4.60751243e-01,
    4.84238370e-01, 9.30276933e-01, 6.54314508e-01, 8.86566453e-01,
    9.98820926e-01, 8.43829346e-01, 9.98926922e-01, 9.46901308e-01,
    9.98877679e-01, 9.90438959e-01, 9.86552834e-01, 9.88188703e-01,
    9.99408619e-01, 9.92567693e-01, 9.69657774e-01, 9.41146828e-01,
    4.19173959e-01, 3.23147520e-03, 3.14583185e-03, 3.19326970e-03,
    9.25592884e-01, 9.50603065e-01, 8.91168802e-01, 7.63411375e-01,
    9.48960923e-01, 9.53967452e-01, 9.59317650e-01, 9.99009419e-01,
    9.94072411e-01, 9.46610316e-01, 9.64551762e-01, 9.62289190e-01,
    9.99305341e-01, 9.99404044e-01, 9.88886844e-01, 9.57556866e-01,
    9.98700397e-01, 9.99340657e-01, 9.98398612e-01, 9.20477165e-01,
    3.16554090e-03, 3.17726399e-03, 1.27931574e-01, 5.64975022e-01,
    9.28232556e-01, 9.79366512e-01, 9.74521398e-01, 8.87847531e-01,
    9.56313577e-01, 9.98910571e-01, 9.82032010e-01, 9.63593149e-01,
    9.99228868e-01, 9.77799176e-01, 9.94789662e-01, 9.91546512e-01,
    9.99229803e-01, 9.45529914e-01, 9.80525063e-01, 9.94226548e-01,
    9.99089073e-01, 9.98984526e-01, 3.15345665e-03, 3.14630284e-03,
    6.00601330e-01, 9.97417268e-01, 3.79870136e-01, 9.78736451e-01,
    9.84562409e-01, 9.66390304e-01, 9.98608145e-01, 9.19031990e-01,
    8.54414855e-01, 9.55057250e-01, 9.99121247e-01, 9.62060345e-01,
    9.59653671e-01, 9.66594663e-01, 9.52281352e-01, 9.48375919e-01,
    9.77709385e-01, 9.92676036e-01, 9.94948857e-01, 9.81449157e-01,
    3.13368592e-03, 3.14118925e-03, 6.53139990e-02, 4.59749667e-01,
    7.51631912e-01, 8.36188902e-01, 9.95072604e-01, 9.87823410e-01,
    9.98650862e-01, 9.84954862e-01, 9.64884363e-01, 9.84953736e-01,
    9.49714955e-01, 9.27425036e-01, 9.38354502e-01, 9.99154603e-01,
    9.64462792e-01, 9.99220744e-01, 9.99288785e-01, 9.99252874e-01,
    9.98562896e-01, 9.77664929e-01};

  std::array<double, 110> constexpr mu_arr = {
    5.11728114e-01,  2.45073138e+00, 4.52629062e+00,  6.16461776e+00,
    8.18578476e+00,  1.05000674e+01, 1.46462484e+01,  1.84203262e+01,
    2.23414249e+01,  2.50654122e+01, 2.87599971e+01,  3.52318878e+01,
    3.92751678e+01,  4.69788706e+01, 5.80937632e+01,  6.99247880e+01,
    8.31551755e+01,  9.71422296e+01, 1.12865609e+02,  1.38626755e+02,
    1.69883109e+02,  1.94200842e+02, 5.71293130e-01,  2.37704254e+00,
    4.47322567e+00,  6.55709447e+00, 8.05667874e+00,  1.08152901e+01,
    1.41102384e+01,  1.81133809e+01, 2.21684299e+01,  2.46358497e+01,
    2.85784801e+01,  3.54666300e+01, 3.87983121e+01,  4.69758809e+01,
    5.85471755e+01,  6.92696255e+01, 8.09117033e+01,  9.26012699e+01,
    1.06587192e+02,  1.32257559e+02, 1.64790423e+02,  1.88804951e+02,
    2.25174129e-01,  2.31498991e+00, 4.19369004e+00,  6.24505066e+00,
    8.04738300e+00,  1.04129959e+01, 1.38767267e+01,  1.87504849e+01,
    2.24702418e+01,  2.51294934e+01, 2.85353636e+01,  3.62322477e+01,
    3.94463100e+01,  4.66744018e+01, 5.85667871e+01,  7.00590244e+01,
    8.27026752e+01,  9.53175091e+01, 1.09882699e+02,  1.35936078e+02,
    1.67337707e+02,  2.07988330e+02, -1.32777123e-01, 1.85008321e+00,
    3.71855544e+00,  5.86765312e+00, 8.24830359e+00,  1.08706965e+01,
    1.38628088e+01,  1.83537022e+01, 2.26605285e+01,  2.52920621e+01,
    2.98566833e+01,  3.66208953e+01, 3.99585050e+01,  4.82400714e+01,
    6.01936452e+01,  7.27421125e+01, 8.44275897e+01,  9.58535278e+01,
    1.06071467e+02,  1.25591863e+02, 1.45716031e+02,  1.69293916e+02,
    -7.82973779e-01, 1.15419569e+00, 3.13718848e+00,  5.31740815e+00,
    6.78591809e+00,  9.92240177e+00, 1.38210054e+01,  1.81819472e+01,
    2.18895890e+01,  2.49283496e+01, 2.89213954e+01,  3.58558894e+01,
    4.00361881e+01,  4.85963348e+01, 6.00490004e+01,  7.16379982e+01,
    8.48249508e+01,  9.56848225e+01, 1.09524905e+02,  1.35669400e+02,
    1.67234082e+02,  1.98382845e+02};

  std::array<double, 110> constexpr tau_arr = {
    3.87497099e+00, 2.87383279e+00, 2.89974546e+00, 8.05299747e-01,
    5.82079679e-01, 4.25342329e-01, 3.16631643e-01, 2.31794166e-01,
    1.87431347e-01, 1.68846430e-01, 1.48897918e-01, 1.23155603e-01,
    1.17395703e-01, 9.65549883e-02, 8.12248716e-02, 6.77644921e-02,
    6.49962883e-02, 7.01815650e-02, 8.45228377e-02, 8.64134443e-02,
    2.47900879e-01, 1.50913981e+00, 3.99000196e+00, 2.99297068e+00,
    2.84870635e+00, 9.02123171e-01, 6.53503944e-01, 3.78202533e-01,
    2.93409763e-01, 2.04011587e-01, 1.70537624e-01, 1.59430876e-01,
    1.35222389e-01, 1.14403226e-01, 9.94072894e-02, 8.86215195e-02,
    7.85378863e-02, 5.96406985e-02, 5.21167369e-02, 4.53305139e-02,
    4.23910526e-02, 4.52286826e-02, 7.12116841e-02, 8.48740567e-02,
    3.99122151e+00, 2.99313873e+00, 1.24303891e+00, 7.65056083e-01,
    5.73145734e-01, 3.80157610e-01, 2.65550203e-01, 2.19892966e-01,
    1.75737759e-01, 1.51955192e-01, 1.29324115e-01, 1.16938673e-01,
    1.01643380e-01, 8.52507286e-02, 7.32086671e-02, 6.14961380e-02,
    5.45104285e-02, 4.75008980e-02, 4.52755127e-02, 4.21972860e-02,
    4.64843157e-02, 1.01169674e-01, 3.88854179e+00, 2.90137190e+00,
    1.09617428e+00, 7.18584529e-01, 5.28846654e-01, 3.61263803e-01,
    2.54901596e-01, 1.89673381e-01, 1.61594277e-01, 1.49463022e-01,
    1.30428339e-01, 1.10958353e-01, 9.64769254e-02, 8.81449968e-02,
    7.75405786e-02, 7.16844329e-02, 6.79409697e-02, 5.76577599e-02,
    4.92408225e-02, 3.67939006e-02, 2.71724239e-02, 2.13570930e-02,
    3.99031276e+00, 2.94701818e+00, 1.24592536e+00, 7.40921522e-01,
    5.08269967e-01, 3.63841999e-01, 2.67205308e-01, 1.92099780e-01,
    1.57416268e-01, 1.48313111e-01, 1.25128868e-01, 1.02699334e-01,
    9.52075683e-02, 8.44257008e-02, 7.10884458e-02, 6.04151374e-02,
    5.41311947e-02, 4.53025254e-02, 4.20917211e-02, 3.90545806e-02,
    5.22777255e-02, 6.61080543e-02};

  // make_short_vec creates an std::vector<double> from an std::array,
  // using all but the last element of the std::array.
  template <size_t N>
  inline std::vector<double>
  make_short_vec(std::array<double, N> const& a)
  {
    static_assert(N != 0, "make_short_vec requires a nonzero-length array");
    return {a.begin(), a.end() - 1};
  }

  // make_vec creates an std::vector<double> from an std::array, using all the
  // values in the std::array.
  template <size_t N>
  inline std::vector<double>
  make_vec(std::array<double, N> const& a)
  {
    return {a.begin(), a.end()};
  }

  // Create an Interp2D from an x-axis, y-axis, and z "matrix", with the matrix
  // unrolled into a one-dimenstional array.
  template <size_t M, std::size_t N>
  inline Interp2D
  make_Interp2D_aux(std::array<double, M> const& xs,
                    std::array<double, N> const& ys,
                    std::array<double, (N - 1) * (M - 1)> const& zs)
  {
    return {make_short_vec(xs), make_short_vec(ys), make_vec(zs)};
  }

  template <size_t M>
  inline Interp2D
  make_Interp2D(std::array<double, M> const& zs)
  {
    return make_Interp2D_aux(lt_bins, zt_bins, zs);
  }
}

Interp2D const LC_LT_t::tau_interp = make_Interp2D(tau_arr);
Interp2D const LC_LT_t::mu_interp = make_Interp2D(mu_arr);
Interp2D const LC_LT_t::sigma_interp = make_Interp2D(sigma_arr);
Interp2D const LC_LT_t::fmsk_interp = make_Interp2D(fmsk_arr);
Interp2D const LC_LT_t::fprj_interp = make_Interp2D(fprj_arr);
