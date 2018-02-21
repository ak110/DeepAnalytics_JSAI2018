# 作業用Makefile (未整理気味)

GPUS?=$(shell python -c 'import pytoolkit as tk ; print(tk.get_gpu_count())')
# SRC?=$(shell dirname `pwd`)
BACKUP_DIR=___history/$(shell cat _current_name)

help:
	@cat Makefile

r1:
	@echo BACKUP_DIR: $(BACKUP_DIR)
	test ! -e models/model.h5
	mpirun -np $(GPUS) python train.py
	python predict.py --no-cache
	cp -rv models $(BACKUP_DIR)
	$(MAKE) val

r1r:
	@echo BACKUP_DIR: $(BACKUP_DIR)
	mpirun -np $(GPUS) python train.py --pseudo-labeling
	python predict.py --no-cache
	cp -rv models $(BACKUP_DIR)
	$(MAKE) val

r2:
	@echo BACKUP_DIR: $(BACKUP_DIR)
	test -e models/model.h5
	mpirun -np $(GPUS) python train.py --warm --pseudo-labeling
	python predict.py --no-cache
	cp -rv models $(BACKUP_DIR)
	$(MAKE) val

r3:
	@echo BACKUP_DIR: $(BACKUP_DIR)
	test -e models/model.h5
	mpirun -np $(GPUS) python train.py --warm --no-validate --pseudo-labeling
	python predict.py --no-cache
	cp -rv models $(BACKUP_DIR)

pred:
	python predict.py --no-cache

val:
	python predict.py --target=val --no-cache --tta-size=64

pip:
	pip-compile requirements.in --output-file docker/requirements.txt

