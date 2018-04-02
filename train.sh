#!/bin/bash
set -x
GPUS=$(nvidia-smi --list-gpus | wc -l)

mkdir -p ___tmp

mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=0 --split-seed=123 
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=0 --split-seed=123 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=1 --split-seed=123
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=1 --split-seed=123 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=2 --split-seed=123
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=2 --split-seed=123 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=3 --split-seed=123
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=3 --split-seed=123 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=4 --split-seed=123
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=4 --split-seed=123 --warm
mv models ___tmp/20180303_InceptionV4_seed123_2

mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=0 --split-seed=234 
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=0 --split-seed=234 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=1 --split-seed=234
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=1 --split-seed=234 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=2 --split-seed=234
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=2 --split-seed=234 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=3 --split-seed=234
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=3 --split-seed=234 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=4 --split-seed=234
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=4 --split-seed=234 --warm
mv models ___tmp/20180303_InceptionV4_seed234_2

mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=0 --split-seed=345 
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=0 --split-seed=345 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=1 --split-seed=345
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=1 --split-seed=345 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=2 --split-seed=345
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=2 --split-seed=345 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=3 --split-seed=345
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=3 --split-seed=345 --warm
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=4 --split-seed=345
mpirun -np $GPUS -H localhost:$(GPUS) python train.py --cv-index=4 --split-seed=345 --warm
mv models ___tmp/20180303_InceptionV4_seed345_2

mkdir models
mv ___tmp/* models/
rmdir ___tmp
