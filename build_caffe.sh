#!/bin/sh

make clean && make superclean
make all -j8 && \
make matcaffe && \
make pycaffe
