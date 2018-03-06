# 作業用Makefile (未整理気味)
GPUS:=$(shell python -c 'import pytoolkit as tk ; print(tk.get_gpu_count())')

help:
	@cat Makefile

train:
	test ! -e models/model.fold0.h5
	test ! -e models/model.fold1.h5
	test ! -e models/model.fold2.h5
	test ! -e models/model.fold3.h5
	test ! -e models/model.fold4.h5
	mpirun -np $(GPUS) -H localhost:4 python train.py --cv-index=0 --split-seed=123
	mpirun -np $(GPUS) -H localhost:4 python train.py --cv-index=1 --split-seed=123
	mpirun -np $(GPUS) -H localhost:4 python train.py --cv-index=2 --split-seed=123
	mpirun -np $(GPUS) -H localhost:4 python train.py --cv-index=3 --split-seed=123
	mpirun -np $(GPUS) -H localhost:4 python train.py --cv-index=4 --split-seed=123

pred:
	rm -rfv models/pred_val
	python predict.py

val:
	rm -rfv models/pred_val
	python predict.py --target=val

clean:
	rm -rfv models/*.log models/proba_val.fold*.pkl models/pred_*
