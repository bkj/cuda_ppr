#!/bin/bash

# run.sh

# --
# Prep data

python prob2bin.py --inpath data/jhu.mtx

# --
# Compile

make clean
make -j12

# --
# Run

./ppr data/jhu.bin > jhu.result