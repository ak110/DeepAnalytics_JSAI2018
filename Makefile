# 作業用Makefile (未整理気味)

help:
	@cat Makefile

pred:
	rm -rfv models/pred_val
	python predict.py

val:
	rm -rfv models/pred_val
	python predict.py --target=val

clean:
	rm -rfv models/pred_* models/*.log

pip:
	pip-compile requirements.in --output-file docker/requirements.txt
