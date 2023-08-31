#!/bin/sh

# This script download the v2p repository, build it,
# and add a symbolic link to the python binding.
#
# It require: cmake, make, and a C99 compiler.

git clone https://github.com/jeremycochoy/v2p.git v2p-repo
cd v2p-repo; mkdir build; cd build; cmake ..; make; cd ../..
ln -s v2p-repo/python/v2p v2p
