# 作業用Makefile (未整理気味)

GPUS?=$(shell python -c 'import pytoolkit as tk ; print(tk.get_gpu_count())')
# SRC?=$(shell dirname `pwd`)

help:
	@cat Makefile

run:
	mpirun -np $(GPUS) -H localhost:$(GPUS) python train.py
	python predict.py --no-cache

predict:
	python predict.py --no-cache

valtta:
	python predict.py --target=val --no-cache

pip:
	pip-compile requirements.in --output-file docker/requirements.txt

