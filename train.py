"""学習。"""
import argparse
import pathlib

import better_exceptions
import horovod.keras as hvd
import numpy as np
import sklearn.externals.joblib as joblib
import sklearn.metrics

import data
import pytoolkit as tk

_MODELS_DIR = pathlib.Path('models')


def _main():
    hvd.init()
    better_exceptions.MAX_LENGTH = 128
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    if hvd.rank() == 0:
        logger.addHandler(tk.log.file_handler(_MODELS_DIR / 'train.log', append=True))
    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        _run()


@tk.log.trace()
def _run():
    import keras
    import models
    logger = tk.log.get(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=300, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=16, type=int)
    parser.add_argument('--warm', help='models/model.fold{cv_index}.h5を読み込む', action='store_true', default=False)
    parser.add_argument('--cv-index', help='CVの何番目か。', type=int)
    parser.add_argument('--cv-size', help='CVの分割数。', default=5, type=int)
    parser.add_argument('--split-seed', help='分割のシード値。', default=123, type=int)
    args = parser.parse_args()
    assert args.cv_index in range(args.cv_size)
    model_path = _MODELS_DIR / 'model.fold{}.h5'.format(args.cv_index)

    (X_train, y_train), (X_val, y_val), _ = data.load_data(args.cv_index, args.cv_size, args.split_seed)
    num_classes = len(np.unique(y_train))
    y_train = tk.ml.to_categorical(num_classes)(y_train)
    y_val = tk.ml.to_categorical(num_classes)(y_val)
    logger.info('len(X_train) = {} len(X_val) = {}'.format(len(X_train), len(X_val)))

    model = models.create_network(num_classes)

    # 学習率：
    # ・lr 0.5、batch size 256くらいが多いのでその辺を基準に
    # ・バッチサイズに比例させるのが良いとのうわさ
    lr = 0.5 * args.batch_size / 256 * hvd.size()
    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(opt, 'categorical_crossentropy', ['acc'])

    if hvd.rank() == 0 and args.cv_index == 0:
        model.summary(print_fn=logger.info)
        logger.info('network depth: %d', tk.dl.count_network_depth(model))

    if args.warm:
        model.load_weights(str(model_path))
        logger.info('{} loaded'.format(model_path))

    callbacks = []
    if args.warm and args.epochs < 300:  # 短縮モード
        callbacks.append(tk.dl.learning_rate_callback((0, 0.5)))
    else:
        callbacks.append(tk.dl.learning_rate_callback())
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
    if hvd.rank() == 0:
        callbacks.append(tk.dl.tsv_log_callback(_MODELS_DIR / 'history.tsv'))
    callbacks.append(tk.dl.freeze_bn_callback(0.95))

    gen = models.create_generator((299, 299), mixup=True)
    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=args.batch_size, data_augmentation=True, shuffle=True),
        steps_per_epoch=gen.steps_per_epoch(len(X_train), args.batch_size) // hvd.size(),
        epochs=args.epochs,
        verbose=1 if hvd.rank() == 0 else 0,
        validation_data=gen.flow(X_val, y_val, batch_size=args.batch_size, shuffle=True),
        validation_steps=gen.steps_per_epoch(len(X_val), args.batch_size) // hvd.size(),  # * 3は省略
        callbacks=callbacks)

    if hvd.rank() == 0:
        model.save(str(model_path))

        proba_val = model.predict_generator(
            gen.flow(X_val, y_val, batch_size=args.batch_size),
            gen.steps_per_epoch(len(X_val), args.batch_size),
            verbose=1)
        joblib.dump(proba_val, _MODELS_DIR / 'proba_val.fold{}.pkl'.format(args.cv_index))

        pred_val = proba_val.argmax(axis=-1)
        logger.info('val_acc: {:.1f}%'.format(sklearn.metrics.accuracy_score(y_val.argmax(axis=-1), pred_val) * 100))


if __name__ == '__main__':
    _main()
