"""学習。"""
import argparse
import multiprocessing
import os
import pathlib

import better_exceptions
import numpy as np
import sklearn.externals.joblib as joblib
import sklearn.metrics

import data
import pytoolkit as tk

_BATCH_SIZE = 32
_MODELS_DIR = pathlib.Path('models')
_RESULT_FORMAT = 'pred_{}/tta{}.pkl'
_TTA0_WEIGHT = 1

_subprocess_context = {}


def _main():
    better_exceptions.MAX_LENGTH = 128
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    logger.addHandler(tk.log.file_handler(_MODELS_DIR / 'predict.log', append=True))
    with tk.dl.session():
        _run()


@tk.log.trace()
def _run():
    logger = tk.log.get(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='GPU数。', type=int, default=tk.get_gpu_count())
    parser.add_argument('--tta-size', help='TTAで何回predictするか。', type=int, default=64)
    parser.add_argument('--target', help='対象のデータ', choices=('val', 'test'), default='test')
    parser.add_argument('--no-cache', help='キャッシュがあれば事前に消す。', action='store_true', default=False)
    args = parser.parse_args()

    # キャッシュの削除
    if args.no_cache:
        for tta_index in range(args.tta_size):
            cache_file = (_MODELS_DIR / _RESULT_FORMAT.format(args.target, tta_index))
            if cache_file.is_file():
                cache_file.unlink()
                logger.info('削除: {}'.format(cache_file))

    # 子プロセスを作って予測
    ctx = multiprocessing.get_context('spawn')
    with multiprocessing.Manager() as m:
        gpu_queue = m.Queue()  # GPUのリスト
        for i in range(args.gpus):
            gpu_queue.put(i)
        with multiprocessing.pool.Pool(args.gpus, _subprocess_init, (gpu_queue, args.target), context=ctx) as pool:  # TODO: プロセスごとのGPUの固定
            pool.starmap(_subprocess, [(args.target, tta_index) for tta_index in range(args.tta_size)])

    # 集計
    pred_target_list = [joblib.load(_MODELS_DIR / _RESULT_FORMAT.format(args.target, tta_index))
                        for tta_index in range(args.tta_size)]
    pred_target_proba = np.average(pred_target_list, axis=0, weights=[_TTA0_WEIGHT] + [1] * (args.tta_size - 1))
    pred_target = pred_target_proba.argmax(axis=-1)

    # 保存
    if args.target == 'test':
        joblib.dump(pred_target_proba, _MODELS_DIR / 'pseudo_label.pkl')
        data.save_data(_MODELS_DIR / 'submit.tsv', pred_target)
    else:
        _, (_, y_val), _ = data.load_data(split=True)
        print('val_acc: {}'.format(sklearn.metrics.accuracy_score(y_val, pred_target)))


def _subprocess_init(gpu_queue, target):
    """子プロセスの初期化。"""
    gpu_id = gpu_queue.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    assert target in ('val', 'test')
    if target == 'val':
        _, (X_target, _), _ = data.load_data(split=True)
    else:
        _, _, X_target = data.load_data()
    _subprocess_context['X_target'] = X_target

    import models
    _subprocess_context['model'] = models.load(_MODELS_DIR / 'model.h5')
    _subprocess_context['gen'] = models.create_generator((256, 256, 3))


def _subprocess(target, tta_index):
    """子プロセスの処理。"""
    result_path = _MODELS_DIR / _RESULT_FORMAT.format(target, tta_index)
    if result_path.is_file():
        print('スキップ: {}'.format(result_path))
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)

        seed = 1234 + tta_index
        np.random.seed(seed)

        X_target = _subprocess_context['X_target']
        model = _subprocess_context['model']
        gen = _subprocess_context['gen']
        da = tta_index != 0

        proba_target = model.predict_generator(
            gen.flow(X_target, batch_size=_BATCH_SIZE, data_augmentation=da, shuffle=False, random_state=seed),
            steps=gen.steps_per_epoch(len(X_target), _BATCH_SIZE),
            verbose=0)

        joblib.dump(proba_target, result_path)
        print('完了: {}'.format(result_path))


if __name__ == '__main__':
    _main()
