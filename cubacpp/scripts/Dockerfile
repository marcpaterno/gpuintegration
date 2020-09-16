#
# Note: this Dockerfile is for use with cosmosis-docker.
#

# This Dockerfile is a bit like a Makefile, except
# that it contains the recipe to build a Docker Image
# (like a virtual machine)

# The parent or base image contains many but not all of the cosmosis
# dependencies environment variables.  See the Dockerfile in base-image
# for how that was created originally (it will be downloaded from 
# hub.docker.io).

FROM joezuntz/cosmosis-base



# OpenBLAS linear algebra installation (provides lapack and blas)
# We could put this in the parent image (joezuntz/cosmosis-test-3) but then
# it would not be tuned as well to your machine.

RUN cd /opt && git clone https://github.com/xianyi/OpenBLAS && cd OpenBLAS && make && make PREFIX=/usr/local install && rm -rf /opt/OpenBLAS

# Many of the python dependencies themselves ultimately depend on LAPACK
# so we need to install them manually here
RUN pip install numpy scipy matplotlib nose pyyaml emcee kombine h5py sklearn cosmolopy astropy



# If you need more libraries or other 
# tools you can put the commands to install them in here, by 
# prefixing the install commands with "RUN".
# Here are some examples:

# You can install with pip:
RUN pip install ipython==5.5.0


# Or with the apt-get command from ubuntu. You can run the VM and search
# inside for what is available like this:
# ./start-cosmosis-vm
# apt-cache search name_to_search_for
#
RUN apt-get install -y cython

# For more involved multi-command installations have a look at the OpenBLAS example
# above that uses && to separate the commands.

# This allows continuous bash history  (so you can press up to get previous commands)
# even when re-launching the machine
ENV HISTFILE /cosmosis/.bash_history

# Install the FITSIO library - this is a pain to do with pip
RUN cd /opt && git clone https://github.com/esheldon/fitsio && cd fitsio && python setup.py build_ext   --use-system-fitsio install && rm -rf /opt/fitsio

RUN pip install future configparser

#Uncomment this if you have issues with unicode docstrings
ENV LANG en_US.UTF-8
ENV PYTHONIOENCODING UTF-8

# If you are on Linux and have problems with permissions then
# you can run on your normal machine:
# id -u  # write down the result and use it as XXX below
# id -g  # write down the result and use it as YYY below
# Then modify and uncomment these lines:
#RUN useradd -r -u XXXX -g YYYY user
#USER user


# Here you can add any of your own dependencies/libraries/etc.
RUN cd /tmp \
  && wget http://www.feynarts.de/cuba/Cuba-4.2.tar.gz \
  && tar xf Cuba-4.2.tar.gz \
  && rm Cuba-4.2.tar.gz \
  && cd Cuba-4.2 \
  && CFLAGS="-O3 -ffast-math -march=native -fomit-frame-pointer -fPIC" ./configure \
  && make lib \
  && make install \
  && cd /usr/local/lib \
  && mkdir /tmp/foo \
  && mv libcuba.a /tmp/foo \
  && cd /tmp/foo \
  && ar xv libcuba.a \
  && gcc -shared *.o -o libcuba.so \
  && rm -f *.o \
  && mv libcuba.so /usr/local/lib \
  && rm -rf /tmp/Cuba-4.2 \
  && rm -rf /tmp/foo
