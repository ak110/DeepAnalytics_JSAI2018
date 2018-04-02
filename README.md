# JSAI2018。

## やったこと

InceptionResNetV2を600epoch回してCVで15個作ってそれぞれTTAを256回やっただけ。

## データの配置

- data
  - test
    - test_0.jpg ～ test_3936.jpg
  - train
    - train_0.jpg ～ train_11994.jpg
  - master.tsv
  - sample_submit.tsv
  - train_master.tsv


## 学習

    cd /path/to/here
    ./train.sh


## 予測

    cd /path/to/here
    docker run --runtime=nvidia --interactive --tty --rm --volume=$PWD:/usr/src/app ak110/keras-docker:0.0.1 python predict.py

