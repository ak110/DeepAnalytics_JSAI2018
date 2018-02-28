"""データの読み書き。"""
import pathlib

import numpy as np
import pandas as pd
import sklearn.model_selection

import pytoolkit as tk

TRAIN_FILE = pathlib.Path('data/train_master.tsv')
SAMPLE_SUBMIT_FILE = pathlib.Path('data/sample_submit.tsv')
CLASS_MASTER_FILE = pathlib.Path('data/master.tsv')

TRAIN_IMAGE_DIR = pathlib.Path('data/train')
TEST_IMAGE_DIR = pathlib.Path('data/test')


@tk.log.trace()
def load_data(cv_index, cv_size, split_seed):
    """データの読み込み。"""
    # 読み込み
    df_train = pd.read_csv(TRAIN_FILE, sep='\t')
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE, sep='\t', header=None)
    X_train = df_train['file_name'].values
    y_train = df_train['category_id'].values
    X_test = df_submit[0].values
    X_train = np.array([TRAIN_IMAGE_DIR / x for x in X_train])
    X_test = np.array([TEST_IMAGE_DIR / x for x in X_test])
    # 分割
    fold = sklearn.model_selection.StratifiedKFold(n_splits=cv_size, shuffle=True, random_state=split_seed)
    train_indices, val_indnces = list(fold.split(X_train, y_train))[cv_index]
    return (X_train[train_indices], y_train[train_indices]), (X_train[val_indnces], y_train[val_indnces]), X_test


@tk.log.trace()
def save_data(path, pred_test):
    """投稿用ファイルの作成。"""
    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE, sep='\t', header=None)
    df_submit[1] = pred_test.astype(int)
    df_submit.to_csv(path, sep='\t', index=False, header=False)


def get_class_names():
    """クラス名を返す。"""
    df_master = pd.read_csv(CLASS_MASTER_FILE, sep='\t', header=None)
    return df_master[0].values
