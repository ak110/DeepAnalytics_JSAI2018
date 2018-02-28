"""予測。"""
import argparse
import multiprocessing
import os
import pathlib

import better_exceptions
import numpy as np
import sklearn.externals.joblib as joblib

import data
import pytoolkit as tk

_BATCH_SIZE = 48
_MODELS_DIR = pathlib.Path('models')
_RESULT_FORMAT = 'pred_{}/fold{}/tta{}.pkl'
_BASE_SIZE = 299
_SIZE_PATTERNS = [
    (int(_BASE_SIZE * 1.25), int(_BASE_SIZE * 1.25)),
    (int(_BASE_SIZE * 1.00), int(_BASE_SIZE * 1.25)),
    (int(_BASE_SIZE * 1.25), int(_BASE_SIZE * 1.00)),
    (int(_BASE_SIZE * 1.00), int(_BASE_SIZE * 1.00)),
    (int(_BASE_SIZE * 1.00), int(_BASE_SIZE * 1.00)),
    (int(_BASE_SIZE * 0.75), int(_BASE_SIZE * 1.00)),
    (int(_BASE_SIZE * 1.00), int(_BASE_SIZE * 0.75)),
    (int(_BASE_SIZE * 0.75), int(_BASE_SIZE * 0.75)),
]

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='GPU数。', type=int, default=tk.get_gpu_count())
    parser.add_argument('--tta-size', help='TTAで何回predictするか。', type=int, default=256)
    parser.add_argument('--cv-size', help='CVの分割数。', default=5, type=int)
    parser.add_argument('--split-seed', help='分割のシード値。', default=123, type=int)
    parser.add_argument('--target', help='対象のデータ', choices=('val', 'test'), default='test')
    parser.add_argument('--no-predict', help='予測を実行しない。', action='store_true', default=False)
    args = parser.parse_args()

    # 子プロセスを作って予測
    if not args.no_predict:
        for cv_index in range(args.cv_size):
            _predict_fold(args, cv_index)

    if args.target == 'test':
        # 集計
        pred_target_list = [[joblib.load(_MODELS_DIR / _RESULT_FORMAT.format(args.target, cv_index, tta_index))
                             for tta_index in range(args.tta_size)]
                            for cv_index in range(args.cv_size)]
        pred_target_proba = np.mean(pred_target_list, axis=(0, 1))
        pred_target = pred_target_proba.argmax(axis=-1)
        # 保存
        joblib.dump(pred_target_proba, _MODELS_DIR / 'pseudo_label.pkl')
        data.save_data(_MODELS_DIR / 'submit.tsv', pred_target)


def _predict_fold(args, cv_index):
    """GPU数分の子プロセスを作って予測を行う。"""
    ctx = multiprocessing.get_context('spawn')
    with multiprocessing.Manager() as m:

        gpu_queue = m.Queue()  # GPUのリスト
        for i in range(args.gpus):
            gpu_queue.put(i)

        with multiprocessing.pool.Pool(args.gpus, _subprocess_init, (gpu_queue, args, cv_index), context=ctx) as pool:
            args_list = [(tta_index, args, cv_index) for tta_index in range(args.tta_size)]
            pool.starmap(_subprocess, args_list, chunksize=1)


def _subprocess_init(gpu_queue, args, cv_index):
    """子プロセスの初期化。"""
    gpu_id = gpu_queue.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import models

    assert args.target in ('val', 'test')
    if args.target == 'val':
        _, (X_target, _), _ = data.load_data(cv_index, args.cv_size, args.split_seed)
    else:
        _, _, X_target = data.load_data(cv_index, args.cv_size, args.split_seed)
    _subprocess_context['X_target'] = X_target

    _subprocess_context['model'] = models.load(_MODELS_DIR / 'model.fold{}.h5'.format(cv_index))


def _subprocess(tta_index, args, cv_index):
    """子プロセスの予測処理。"""
    import models

    result_path = _MODELS_DIR / _RESULT_FORMAT.format(args.target, cv_index, tta_index)
    if result_path.is_file():
        print('スキップ: {}'.format(result_path))
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)

        seed = 1234 + tta_index
        np.random.seed(seed)

        pattern_index = len(_SIZE_PATTERNS) * tta_index // args.tta_size
        img_size = _SIZE_PATTERNS[pattern_index]
        batch_size = int(_BATCH_SIZE * ((_BASE_SIZE ** 2) / (img_size[0] * img_size[1])) ** 1.5)
        gen = models.create_generator(img_size, mixup=False)

        X_target = _subprocess_context['X_target']
        model = _subprocess_context['model']

        proba_target = model.predict_generator(
            gen.flow(X_target, batch_size=batch_size, data_augmentation=True, shuffle=False, random_state=seed),
            steps=gen.steps_per_epoch(len(X_target), batch_size),
            verbose=0)

        joblib.dump(proba_target, result_path)
        print('完了: {}'.format(result_path))


if __name__ == '__main__':
    _main()
