"""学習。"""
import argparse
import pathlib

import better_exceptions
import horovod.keras as hvd
import numpy as np
import sklearn.model_selection

import data
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')


def _main():
    hvd.init()
    better_exceptions.MAX_LENGTH = 128
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    logger.addHandler(tk.log.file_handler(MODELS_DIR / 'train.log'))
    with tk.dl.session(gpu_options={'visible_device_list': str(hvd.local_rank())}):
        _run()


@tk.log.trace()
def _run():
    import keras
    import models

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='epoch数。', default=300, type=int)
    parser.add_argument('--batch-size', help='バッチサイズ。', default=32, type=int)
    parser.add_argument('--no-validate', help='バリデーションしない。', action='store_true', default=False)
    parser.add_argument('--warm', help='models/model.h5を読み込む', action='store_true', default=False)
    args = parser.parse_args()
    validate = not args.no_validate

    (X_train, y_train), _ = data.load_data()
    if validate:
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            X_train, y_train, test_size=0.1, random_state=123)
    else:
        X_val, y_val = None, None
    num_classes = len(np.unique(y_train))

    model, input_shape = models.create_network(num_classes)

    # 学習率：
    # ・lr 0.5、batch size 256くらいが多いのでその辺を基準に
    # ・バッチサイズに比例させるのが良いとのうわさ
    lr = 0.5 * args.batch_size / 256 * hvd.size()
    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(opt, 'categorical_crossentropy', ['acc'])

    if hvd.rank() == 0:
        model.summary(print_fn=tk.log.get(__name__).info)
        tk.log.get(__name__).info('network depth: %d', tk.dl.count_network_depth(model))

    if args.warm:
        model.load_weights('models/model.h5')
        tk.log.get(__name__).info('models/model.h5 loaded')

    callbacks = []
    callbacks.append(tk.dl.learning_rate_callback(lr=lr, epochs=args.epochs))
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
    if hvd.rank() == 0:
        callbacks.append(tk.dl.tsv_log_callback(MODELS_DIR / 'history.tsv'))

    gen = models.create_generator(input_shape, num_classes)
    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=args.batch_size, data_augmentation=True, shuffle=True),
        steps_per_epoch=gen.steps_per_epoch(len(X_train), args.batch_size) // hvd.size(),
        epochs=args.epochs,
        verbose=1 if hvd.rank() == 0 else 0,
        validation_data=gen.flow(X_val, y_val, batch_size=args.batch_size, shuffle=True) if validate else None,
        validation_steps=gen.steps_per_epoch(len(X_val), args.batch_size) // hvd.size() if validate else None,  # * 3は省略
        callbacks=callbacks)

    if hvd.rank() == 0:
        model.save(str(MODELS_DIR / 'model.h5'))


if __name__ == '__main__':
    _main()
