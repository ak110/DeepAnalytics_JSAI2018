"""データの読み書き。"""
import pathlib

import pandas as pd

import pytoolkit as tk

TRAIN_FILE = pathlib.Path('data/train_master.tsv')
SAMPLE_SUBMIT_FILE = pathlib.Path('data/sample_submit.tsv')


@tk.log.trace()
def load_data():
    """データの読み込み。"""
    df_train = pd.read_csv(TRAIN_FILE, sep='\t')
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE, sep='\t', header=None)
    X_train = df_train['file_name'].values
    y_train = df_train['category_id'].values
    X_test = df_submit[0].values
    return (X_train, y_train), X_test


@tk.log.trace()
def save_data(path, pred_test):
    """投稿用ファイルの作成。"""
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE, sep='\t', header=None)
    df_submit[1] = pred_test.astype(int)
    df_submit.to_csv(path, sep='\t', index=False, header=False)
