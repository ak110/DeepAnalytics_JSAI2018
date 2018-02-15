#!/bin/bash
set -eux
GPUS=$(python -c 'import pytoolkit as tk ; print(tk.get_gpu_count())')
mpirun -np $GPUS -H localhost:$GPUS python train.py $*
python predict.py --gpus=$GPUS
