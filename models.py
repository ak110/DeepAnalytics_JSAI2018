"""モデル。"""

import pytoolkit as tk


@tk.log.trace()
def create_network(num_classes: int):
    """ネットワークを作って返す。"""
    import keras

    input_shape = (256, 256, 3)

    base_model = keras.applications.DenseNet201(include_top=False, weights=None, input_shape=input_shape)
    x = base_model.outputs[0]
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='zeros', name='predictions')(x)

    model = keras.models.Model(inputs=base_model.inputs, outputs=x)
    return model, input_shape


def create_generator(input_shape, num_classes):
    """ImageDataGeneratorを作って返す。"""
    gen = tk.image.ImageDataGenerator(
        input_shape[:2], label_encoder=tk.ml.to_categorical(num_classes),
        rotate_degrees=180)
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.5, tk.image.RandomErasing())
    gen.add(0.25, tk.image.RandomBlur())
    gen.add(0.25, tk.image.RandomBlur(partial=True))
    gen.add(0.25, tk.image.RandomUnsharpMask())
    gen.add(0.25, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.25, tk.image.RandomMedian())
    gen.add(0.25, tk.image.GaussianNoise())
    gen.add(0.25, tk.image.GaussianNoise(partial=True))
    gen.add(0.5, tk.image.RandomSaturation())
    gen.add(0.5, tk.image.RandomBrightness())
    gen.add(0.5, tk.image.RandomContrast())
    gen.add(0.5, tk.image.RandomHue())
    return gen
