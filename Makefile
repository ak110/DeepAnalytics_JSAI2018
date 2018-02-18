# 作業用Makefile (未整理気味)

GPUS?=$(shell python -c 'import pytoolkit as tk ; print(tk.get_gpu_count())')
# SRC?=$(shell dirname `pwd`)
BACKUP_DIR=___history/20180216_pseudo_labeling

help:
	@cat Makefile

r1:
	@echo BACKUP_DIR: $(BACKUP_DIR)
	test ! -e models/model.h5
	mpirun -np $(GPUS) -H localhost:$(GPUS) python train.py
	python predict.py --no-cache
	cp -rv models $(BACKUP_DIR)

r2:
	@echo BACKUP_DIR: $(BACKUP_DIR)
	test -e models/model.h5
	mpirun -np $(GPUS) -H localhost:$(GPUS) python train.py --warm --no-validate --pseudo-labeling
	python predict.py --no-cache
	cp -rv models $(BACKUP_DIR)

pred:
	python predict.py --no-cache

val:
	python predict.py --target=val --no-cache

val16:
	python predict.py --target=val --no-cache --tta-size=16

pip:
	pip-compile requirements.in --output-file docker/requirements.txt

