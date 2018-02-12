#!/bin/bash
set -eux
mpirun -np 2 -H localhost:2 python train.py --warm
python predict.py
