"""モデル。"""

import pytoolkit as tk


@tk.log.trace()
def create_network(num_classes: int):
    """ネットワークを作って返す。"""
    import keras

    input_shape = (256, 256, 3)

    builder = tk.dl.Builder()
    builder.set_default_l2()
    builder.conv_defaults['kernel_initializer'] = 'he_uniform'

    def _block(x, filters, res_count, name):
        for res in range(res_count):
            sc = x
            x = builder.conv2d(filters, (3, 3), name='{}_r{}c1'.format(name, res))(x)
            x = keras.layers.Dropout(0.25)(x)
            x = builder.conv2d(filters, (3, 3), use_act=False, name='{}_r{}c2'.format(name, res))(x)
            x = keras.layers.add([sc, x])
        x = builder.bn()(x)
        x = builder.act()(x)
        return x

    def _tran(x, filters, name):
        x = builder.conv2d(filters, (1, 1), name='{}_conv'.format(name))(x)
        x = keras.layers.MaxPooling2D()(x)
        return x

    x = inp = keras.layers.Input(input_shape)
    x = builder.conv2d(64, (7, 7), strides=(2, 2), name='start')(x)  # 128
    x = keras.layers.MaxPooling2D()(x)  # 64
    x = _block(x, 64, 4, name='stage1_block')
    x = _tran(x, 128, name='stage1_tran')  # 32
    x = _block(x, 128, 4, name='stage2_block')
    x = _tran(x, 256, name='stage2_tran')  # 16
    x = _block(x, 256, 4, name='stage3_block')
    x = _tran(x, 512, name='stage3_tran')  # 8
    x = _block(x, 512, 4, name='stage4_block')
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(num_classes, activation='softmax', kernel_initializer='zeros', name='predictions')(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    return model, input_shape


def create_generator(input_shape, num_classes):
    """ImageDataGeneratorを作って返す。"""
    gen = tk.image.ImageDataGenerator(
        input_shape[:2], label_encoder=tk.ml.to_categorical(num_classes),
        rotate_degrees=180)
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.5, tk.image.FlipTB())
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
