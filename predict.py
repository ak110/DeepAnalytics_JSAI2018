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

_BATCH_SIZE = 48
_MODELS_DIR = pathlib.Path('models')
_RESULT_FORMAT = 'pred_{}/tta{}.pkl'
_BASE_SIZE = 299
_SIZE_PATTERNS = [
    (int(_BASE_SIZE * 1.50), int(_BASE_SIZE * 1.50)),
    (int(_BASE_SIZE * 1.25), int(_BASE_SIZE * 1.25)),

    (int(_BASE_SIZE * 1.00), int(_BASE_SIZE * 1.25)),
    (int(_BASE_SIZE * 1.25), int(_BASE_SIZE * 1.00)),

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
    logger = tk.log.get(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', help='GPU数。', type=int, default=tk.get_gpu_count())
    parser.add_argument('--tta-size', help='TTAで何回predictするか。', type=int, default=256)
    parser.add_argument('--target', help='対象のデータ', choices=('val', 'test'), default='test')
    parser.add_argument('--no-predict', help='予測を実行しない。', action='store_true', default=False)
    args = parser.parse_args()

    # 子プロセスを作って予測
    if not args.no_predict:
        ctx = multiprocessing.get_context('spawn')
        with multiprocessing.Manager() as m:
            gpu_queue = m.Queue()  # GPUのリスト
            for i in range(args.gpus):
                gpu_queue.put(i)
            with multiprocessing.pool.Pool(args.gpus, _subprocess_init, (gpu_queue, args.target), context=ctx) as pool:
                args_list = [(args.target, tta_index, args.tta_size) for tta_index in range(args.tta_size)]
                pool.starmap(_subprocess, args_list, chunksize=1)

    # 集計
    pred_target_list = [joblib.load(_MODELS_DIR / _RESULT_FORMAT.format(args.target, tta_index))
                        for tta_index in range(args.tta_size)]
    pred_target_proba = np.mean(pred_target_list, axis=0)
    pred_target = pred_target_proba.argmax(axis=-1)

    if args.target == 'test':
        # 保存
        joblib.dump(pred_target_proba, _MODELS_DIR / 'pseudo_label.pkl')
        data.save_data(_MODELS_DIR / 'submit.tsv', pred_target)
    else:
        _, (X_val, y_val), _ = data.load_data(split=True)
        # 正解率
        logger.info('val_acc: {}'.format(sklearn.metrics.accuracy_score(y_val, pred_target)))
        # classification_report
        class_names = data.get_class_names()
        logger.info(sklearn.metrics.classification_report(y_val, pred_target, target_names=class_names))
        # 個別の結果 (今回は間違ってるのが少ないので全部出しちゃう)
        for path, pred, label in sorted(zip(X_val, pred_target, y_val)):
            if pred != label:
                logger.info('{}: pred={} label={}'.format(path.name, class_names[pred], class_names[label]))
        # サイズパターンごとの結果
        pattern_indices = []
        for tta_index in range(args.tta_size):
            pat_index = len(_SIZE_PATTERNS) * tta_index // args.tta_size
            if len(pattern_indices) <= pat_index:
                pattern_indices.append([])
            pattern_indices[pat_index].append(tta_index)
        for img_size, pat_indices in zip(_SIZE_PATTERNS, pattern_indices):
            pat_data = [pred_target_list[pi] for pi in pat_indices]
            mean_acc = np.mean([sklearn.metrics.accuracy_score(y_val, d.argmax(axis=-1)) for d in pat_data])  # 個別のaccuracyの平均
            mix_acc = sklearn.metrics.accuracy_score(y_val, np.mean(pat_data, axis=0).argmax(axis=-1))  # 当該サイズだけでmeanした後の正解率
            logger.info('size={}: mean acc={:.3f} mix acc={:.3f}'.format(img_size, mean_acc, mix_acc))


def _subprocess_init(gpu_queue, target):
    """子プロセスの初期化。"""
    gpu_id = gpu_queue.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import models

    assert target in ('val', 'test')
    if target == 'val':
        _, (X_target, _), _ = data.load_data(split=True)
    else:
        _, _, X_target = data.load_data()
    _subprocess_context['X_target'] = X_target

    _subprocess_context['model'] = models.load(_MODELS_DIR / 'model.h5')


def _subprocess(target, tta_index, tta_size):
    """子プロセスの処理。"""
    import models

    result_path = _MODELS_DIR / _RESULT_FORMAT.format(target, tta_index)
    if result_path.is_file():
        print('スキップ: {}'.format(result_path))
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)

        seed = 1234 + tta_index
        np.random.seed(seed)

        pattern_index = len(_SIZE_PATTERNS) * tta_index // tta_size
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
