"""データの読み書き。"""
import pathlib

import numpy as np
import pandas as pd

import pytoolkit as tk

TRAIN_FILE = pathlib.Path('data/train_master.tsv')
SAMPLE_SUBMIT_FILE = pathlib.Path('data/sample_submit.tsv')

TRAIN_IMAGE_DIR = pathlib.Path('data/train')
TEST_IMAGE_DIR = pathlib.Path('data/test')


@tk.log.trace()
def load_data():
    """データの読み込み。"""
    df_train = pd.read_csv(TRAIN_FILE, sep='\t')
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE, sep='\t', header=None)
    X_train = df_train['file_name'].values
    y_train = df_train['category_id'].values
    X_test = df_submit[0].values
    X_train = np.array([TRAIN_IMAGE_DIR / x for x in X_train])
    X_test = np.array([TEST_IMAGE_DIR / x for x in X_test])
    return (X_train, y_train), X_test


@tk.log.trace()
def save_data(path, pred_test):
    """投稿用ファイルの作成。"""
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE, sep='\t', header=None)
    df_submit[1] = pred_test.astype(int)
    df_submit.to_csv(path, sep='\t', index=False, header=False)
