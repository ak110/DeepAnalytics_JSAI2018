"""予測。"""
import argparse
import concurrent.futures
import multiprocessing
import os
import pathlib

import better_exceptions
import numpy as np
import sklearn.externals.joblib as joblib
import tqdm

import data
import pytoolkit as tk

_BATCH_SIZE = 48
_MODELS_DIR = pathlib.Path('models')
_CACHE_NAME_FORMAT = 'pred_{target}_{model}/tta{tta}.pkl'
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
_MODELS = [
    # models_dir, model_name, cv_index, cv_size, split_seed
    (_MODELS_DIR / '20180303_InceptionV4_seed123_2', 'model.fold0.h5', 0, 5, 123),
    (_MODELS_DIR / '20180303_InceptionV4_seed123_2', 'model.fold1.h5', 1, 5, 123),
    (_MODELS_DIR / '20180303_InceptionV4_seed123_2', 'model.fold2.h5', 2, 5, 123),
    (_MODELS_DIR / '20180303_InceptionV4_seed123_2', 'model.fold3.h5', 3, 5, 123),
    (_MODELS_DIR / '20180303_InceptionV4_seed123_2', 'model.fold4.h5', 4, 5, 123),
    (_MODELS_DIR / '20180303_InceptionV4_seed234_2', 'model.fold0.h5', 0, 5, 234),
    (_MODELS_DIR / '20180303_InceptionV4_seed234_2', 'model.fold1.h5', 1, 5, 234),
    (_MODELS_DIR / '20180303_InceptionV4_seed234_2', 'model.fold2.h5', 2, 5, 234),
    (_MODELS_DIR / '20180303_InceptionV4_seed234_2', 'model.fold3.h5', 3, 5, 234),
    (_MODELS_DIR / '20180303_InceptionV4_seed234_2', 'model.fold4.h5', 4, 5, 234),
    (_MODELS_DIR / '20180303_InceptionV4_seed345_2', 'model.fold0.h5', 0, 5, 345),
    (_MODELS_DIR / '20180303_InceptionV4_seed345_2', 'model.fold1.h5', 1, 5, 345),
    (_MODELS_DIR / '20180303_InceptionV4_seed345_2', 'model.fold2.h5', 2, 5, 345),
    (_MODELS_DIR / '20180303_InceptionV4_seed345_2', 'model.fold3.h5', 3, 5, 345),
    (_MODELS_DIR / '20180303_InceptionV4_seed345_2', 'model.fold4.h5', 4, 5, 345),
]


def _main():
    better_exceptions.MAX_LENGTH = 128
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    logger.addHandler(tk.log.file_handler(_MODELS_DIR / 'predict.log', append=True))
    _run()


@tk.log.trace()
def _run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='GPU数。', type=int, default=tk.get_gpu_count())
    parser.add_argument('--tta-size', help='TTAで何回predictするか。', type=int, default=256)
    parser.add_argument('--target', help='対象のデータ', choices=('val', 'test'), default='test')
    parser.add_argument('--no-predict', help='予測を実行しない。', action='store_true', default=False)
    args = parser.parse_args()

    # 子プロセスを作って予測
    for models_dir, model_name, cv_index, cv_size, split_seed in tqdm.tqdm(_MODELS, desc='models', ncols=100, ascii=True):
        _predict_model(args, models_dir, model_name, cv_index, cv_size, split_seed)

    if args.target == 'test':
        # 集計
        pred_target_list = [[joblib.load(model_dir / _CACHE_NAME_FORMAT.format(target=args.target, model=model_name, tta=tta_index))
                             for tta_index in range(args.tta_size)]
                            for model_dir, model_name, cv_index, _, _ in _MODELS]
        pred_target_proba = np.mean(pred_target_list, axis=(0, 1))
        pred_target = pred_target_proba.argmax(axis=-1)
        # 保存
        joblib.dump(pred_target_proba, _MODELS_DIR / 'pseudo_label.pkl')
        data.save_data(_MODELS_DIR / 'submit.tsv', pred_target)


def _predict_model(args, models_dir, model_name, cv_index, cv_size, split_seed):
    """GPU数分の子プロセスを作って予測を行う。"""
    with multiprocessing.Manager() as m, concurrent.futures.ProcessPoolExecutor(args.gpus) as pool:

        gpu_queue = m.Queue()  # GPUのリスト
        for i in range(args.gpus):
            gpu_queue.put(i)

        futures = [pool.submit(_subprocess, gpu_queue, args, models_dir, model_name, cv_index, cv_size, split_seed, tta_index)
                   for tta_index in range(args.tta_size)]
        result_set = {f.result() for f in tqdm.tqdm(futures, desc='tta', ncols=100, ascii=True)}

        logger = tk.log.get(__name__)
        for i, r in enumerate(result_set):
            logger.debug('subprocess output#{}: {}'.format(i, r))


_subprocess_context = {}


@tk.capture_output()
def _subprocess(gpu_queue, args, models_dir, model_name, cv_index, cv_size, split_seed, tta_index):
    """子プロセスの予測処理。"""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # GPUの固定
        gpu_id = gpu_queue.get()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # TensorFlowのログ抑止
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # モデルの読み込み
        import models
        _subprocess_context['model'] = models.load(models_dir / model_name)
    else:
        import models

    result_path = pathlib.Path(models_dir / _CACHE_NAME_FORMAT.format(target=args.target, model=model_name, tta=tta_index))
    if result_path.is_file():
        print('スキップ: {}'.format(result_path))
        return
    else:
        seed = 1234 + tta_index
        np.random.seed(seed)

        assert args.target in ('val', 'test')
        if args.target == 'val':
            _, (X_target, _), _ = data.load_data(cv_index, cv_size, split_seed)
        else:
            _, _, X_target = data.load_data(cv_index, cv_size, split_seed)

        pattern_index = len(_SIZE_PATTERNS) * tta_index // args.tta_size
        img_size = _SIZE_PATTERNS[pattern_index]
        batch_size = int(_BATCH_SIZE * ((_BASE_SIZE ** 2) / (img_size[0] * img_size[1])) ** 1.5)

        gen = models.create_generator(img_size, mixup=False)
        proba_target = _subprocess_context['model'].predict_generator(
            gen.flow(X_target, batch_size=batch_size, data_augmentation=True, shuffle=False, random_state=seed),
            steps=gen.steps_per_epoch(len(X_target), batch_size),
            verbose=0)

        result_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(proba_target, result_path)


if __name__ == '__main__':
    _main()
