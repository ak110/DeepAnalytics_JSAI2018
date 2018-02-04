"""学習。"""
import pathlib

import better_exceptions
import numpy as np

import data
import pytoolkit as tk

BATCH_SIZE = 32
MODELS_DIR = pathlib.Path('models')


def _main():
    better_exceptions.MAX_LENGTH = 128
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger = tk.log.get()
    logger.addHandler(tk.log.stream_handler())
    logger.addHandler(tk.log.file_handler(MODELS_DIR / 'predict.log', append=True))
    with tk.dl.session():
        _run()


@tk.log.trace()
def _run():
    import models

    (_, y_train), X_test = data.load_data()
    num_classes = len(np.unique(y_train))

    model, input_shape = models.create_network(num_classes)
    gen = models.create_generator(input_shape, num_classes)

    pred_test = model.predict_generator(
        gen.flow(X_test, batch_size=BATCH_SIZE, data_augmentation=False, shuffle=False),
        steps=gen.steps_per_epoch(len(X_test), BATCH_SIZE),
        verbose=1)

    data.save_data(MODELS_DIR / 'submit.tsv', pred_test)


if __name__ == '__main__':
    _main()
