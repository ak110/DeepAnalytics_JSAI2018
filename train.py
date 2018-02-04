"""学習。"""
import pathlib

import better_exceptions
import horovod.keras as hvd
import numpy as np

import data
import pytoolkit as tk

BATCH_SIZE = 32
MAX_EPOCH = 10
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

    (X_train, y_train), _ = data.load_data()
    num_classes = len(np.unique(y_train))

    model, input_shape = models.create_network(num_classes)

    # 学習率：
    # ・lr 0.5、batch size 256くらいが多いのでその辺を基準に
    # ・バッチサイズに比例させるのが良いとのうわさ
    lr = 0.5 * BATCH_SIZE / 256 * hvd.size()
    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(opt, 'categorical_crossentropy', ['acc'])

    if hvd.rank() == 0:
        model.summary(print_fn=tk.log.get(__name__).info)
        tk.log.get(__name__).info('network depth: %d', tk.dl.count_network_depth(model))

    callbacks = []
    callbacks.append(tk.dl.learning_rate_callback(lr=lr, epochs=MAX_EPOCH))
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
    if hvd.rank() == 0:
        callbacks.append(tk.dl.tsv_log_callback(MODELS_DIR / 'history.tsv'))

    gen = models.create_generator(input_shape, num_classes)
    model.fit_generator(
        gen.flow(X_train, y_train, batch_size=BATCH_SIZE, data_augmentation=True, shuffle=True),
        steps_per_epoch=gen.steps_per_epoch(X_train.shape[0], BATCH_SIZE) // hvd.size(),
        epochs=MAX_EPOCH,
        verbose=1 if hvd.rank() == 0 else 0,
        # validation_data=gen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=True),
        # validation_steps=gen.steps_per_epoch(X_test.shape[0], BATCH_SIZE) // hvd.size(),  # * 3は省略
        callbacks=callbacks)

    if hvd.rank() == 0:
        model.save(str(MODELS_DIR / 'model.h5'))


if __name__ == '__main__':
    _main()
