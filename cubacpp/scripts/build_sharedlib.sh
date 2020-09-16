# Download the CUBA library, and build it into a shared library.
#
# Updates to the CUBA library release version will require an update to
# CUVA_VERSION.

die()
{
  local exitval
  if [[ "$1" =~ ^[0-9]*$ ]]; then (( exitval = $1 )); shift; fi
  printf "ERROR: $@\n" 1>&2
  exit ${exitval:-1}
}

CUBA_VERSION=4.2
CUBA_TARBALL=Cuba-${CUBA_VERSION}

INSTALL_DIR=${INSTALL_DIR:-$HOME}
export CFLAGS="-O3 -ffast-math -fomit-frame-pointer -march=native -fPIC"
CC=gcc

workdir=$(mktemp -d -t XXXXXXXXXX) || die "Failed to create temporary directory"
echo "Working directory is ${workdir}"
pushd ${workdir} || die "Failed to cd to working directory"
wget http://www.feynarts.de/cuba/${CUBA_TARBALL}.tar.gz || die "Failed to download source"
tar xf ${CUBA_TARBALL}.tar.gz || die "Failed to untar source"
cd ${CUBA_TARBALL} || die "Failed to move to ${workdir}/${CUBA_TARBALL}"
./configure --prefix ${INSTALL_DIR}|| die "Failed to configure CUDA build"
make lib || die "Failed to build static library"
make install || die "Failed to install static library"
cd ${INSTALL_DIR}/lib || die "Failed to move to ${INSTALL_DIR}/lib"
ar xv libcuba.a || die "Failed to unpack static library"
${CC} -shared *.o -o libcuba.so || die "Failed to build shared library"
rm -f "__.SYMDEF SORTED" {,ll}{Vegas,Suave,Divonne,Cuhre}{,_}.o {Fork,Global}{,_}.o Data.o libcuba.a || die "Failed to remove stray files"
popd
if [[ -d "${WORKDIR}" ]]; then
    rm -r "${WORKDIR}" || die "Failed to clean up workdir ${WORKDIR}"
fi
echo "CUBA has been installed in ${INSTALL_DIR}"
