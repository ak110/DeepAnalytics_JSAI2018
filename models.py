"""モデル。"""

import pytoolkit as tk


@tk.log.trace()
def create_network(num_classes: int):
    """ネットワークを作って返す。"""
    import keras
    base_model = keras.applications.InceptionResNetV2(include_top=False, weights=None, input_shape=(None, None, 3))
    x = base_model.outputs[0]
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='zeros', name='predictions')(x)
    model = keras.models.Model(inputs=base_model.inputs, outputs=x)
    return model


def load(path):
    """読み込んで返す。"""
    import keras
    return keras.models.load_model(str(path), compile=False)


def create_generator(img_size, mixup):
    """ImageDataGeneratorを作って返す。"""
    gen = tk.image.ImageDataGenerator()
    # gen.add(tk.image.ProcessOutput(tk.ml.to_categorical(num_classes), batch_axis=True))
    gen.add(tk.image.Resize(img_size))
    if mixup:
        gen.add(tk.image.Mixup(probability=1))
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5, degrees=180))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize(img_size))
    gen.add(tk.image.RandomFlipLRTB(probability=0.5))
    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.5),
        tk.image.RandomUnsharpMask(probability=0.5),
        tk.image.GaussianNoise(probability=0.5),
        tk.image.RandomSaturation(probability=0.5),
        tk.image.RandomBrightness(probability=0.5),
        tk.image.RandomContrast(probability=0.5),
        tk.image.RandomHue(probability=0.5),
    ]))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.ProcessInput(tk.image.preprocess_input_abs1))  # 転移学習しないのでkeras.applications.densenet.preprocess_inputである必要は無い
    return gen
