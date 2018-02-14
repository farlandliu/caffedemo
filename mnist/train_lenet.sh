#!/usr/bin/env sh
# export PATH=/home/farland/caffe/build/tools:$PATH

set -e

caffe train --solver=lenet_solver.prototxt --snapshot=lenet_train_farland__iter_3434.solverstate $@
