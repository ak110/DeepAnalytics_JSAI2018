"""学習。"""
import argparse
import multiprocessing
import pathlib

import better_exceptions
import numpy as np
import sklearn.externals.joblib as joblib

import data
import pytoolkit as tk

_BATCH_SIZE = 32
_MODELS_DIR = pathlib.Path('models')
_TTA0_WEIGHT = 20  # TODO: 調整


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
    parser.add_argument('--tta-size', help='TTAで何回predictするか。', type=int, default=64)
    parser.add_argument('--target', help='対象のデータ', choises=('val', 'test'), default='test')
    args = parser.parse_args()

    # GPU数分の子プロセスを作って予測
    gpu_queue = multiprocessing.Queue(maxsize=args.gpus)  # 空いてるGPUのリスト
    for i in range(args.gpus):
        gpu_queue.put(i)
    with multiprocessing.Pool(args.gpus, maxtasksperchild=1) as pool:
        pool.starmap(_subprocess, [(args.target, tta_index, gpu_queue) for tta_index in range(args.tta_size)])

    # 集計
    pred_target_list = [joblib.load(_MODELS_DIR / 'pred_{}_tta{}.pkl'.format(args.target, tta_index))
                        for tta_index in range(args.tta_size)]
    pred_target_proba = np.average(pred_target_list, axis=0, weights=[_TTA0_WEIGHT] + [1] * (args.tta_size - 1))
    pred_target = pred_target_proba.argmax(axis=-1)

    # 保存
    data.save_data(_MODELS_DIR / 'submit.tsv', pred_target)


def _subprocess(target, tta_index, gpu_queue):
    gpu_id = gpu_queue.get()
    try:
        with tk.dl.session(gpu_options={'visible_device_list': str(gpu_id)}):
            _subprocess2(target, tta_index)
    finally:
        gpu_queue.put(gpu_id)


def _subprocess2(target, tta_index):
    import models

    assert target in ('val', 'test')
    if target == 'val':
        _, (X_target, _), _ = data.load_data(split=True)
    else:
        _, _, X_target = data.load_data()

    model = models.load(_MODELS_DIR / 'model.h5')
    gen = models.create_generator((256, 256, 3))

    data_augmentation = tta_index != 0
    pred_target_proba = model.predict_generator(
        gen.flow(X_target, batch_size=_BATCH_SIZE, data_augmentation=data_augmentation, shuffle=False),
        steps=gen.steps_per_epoch(len(X_target), _BATCH_SIZE),
        verbose=1)

    joblib.dump(pred_target_proba, _MODELS_DIR / 'pred_{}_tta{}.pkl'.format(target, tta_index))


if __name__ == '__main__':
    _main()
