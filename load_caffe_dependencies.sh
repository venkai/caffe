#!/bin/sh

# This file will be automatically sourced before running any of the training
# or inference scripts. You can use this to load dependencies such as cuda/hdf5 
# which are often installed in non-standard locations by using modules and/or
# manually adding their library paths to LD_LIBRARY_PATH
# Please see ./load_caffe_dependencies.sh.example

echo "Loading dependencies for running/compiling caffe"

# Specify path to your MATLAB installation that you used in Makefile.config
# This will be used for running matcaffe inference scripts
MATLAB_DIR=/opt/common/matlab-r2014b
export MATLAB_DIR

# Do not modify below
# ---------------------
HDF5_DISABLE_VERSION_CHECK=2
export HDF5_DISABLE_VERSION_CHECK

echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
echo 
echo done...