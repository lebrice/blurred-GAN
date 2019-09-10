import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import blurred_gan
from blurred_gan import WGANGP, AdaptiveBlurController, BlurScheduleController

import utils


def dataset(shuffle_buffer_size=256) -> tf.data.Dataset:
    """Modern Tensorflow input pipeline for the CelebA dataset"""

    @tf.function
    def take_image(example):
        return example["image"]

    @tf.function
    def convert_to_float(image):
        return (tf.cast(image, tf.float32) - 127.5) / 127.5

    @tf.function
    def preprocess_images(image):
        image = convert_to_float(image)
        return image

    dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)

    dataset = (dataset
        .map(take_image)
        .batch(16)  # make preprocessing faster by batching inputs.
        .map(preprocess_images)
        .apply(tf.data.experimental.unbatch())
    #   .cache("./cache/")
        .shuffle(shuffle_buffer_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    return dataset


class Generator(tf.keras.Sequential):
    def __init__(self, latent_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_size = latent_size

        self.add(layers.Dense(7*7*256, use_bias=False, input_shape=(self.latent_size,)))
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Reshape((7, 7, 256)))
        assert self.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        self.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.output_shape == (None, 7, 7, 128)
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.output_shape == (None, 14, 14, 64)
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.output_shape == (None, 28, 28, 1)


class Discriminator(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        self.add(layers.LeakyReLU())
        self.add(layers.Dropout(0.3))

        self.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.add(layers.LeakyReLU())
        self.add(layers.Dropout(0.3))

        self.add(layers.Flatten())
        self.add(layers.Dense(1))


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import datetime

    epochs = 10
    batch_size = 256



    # we add a useless 'label' for the fit method to work.
    dataset = dataset().batch(batch_size).map(lambda v: (v, 0))

    from gaussian_blur import maximum_reasonable_std

    # path = utils.locate_model_file("results", "mnist")
    # if path is not None:
    #     print(f"Loading weights from '{path}'")
    #     log_dir = "/".join(path.split("/")[:-1])

    #     gan = tf.keras.models.load_model(path)
    #     initial_epoch = utils.epoch(path)
    # else:

    print("Starting from scratch.")
    log_dir = utils.create_result_subdir("results", "mnist")

    gen = Generator()
    disc = Discriminator()
    gan = WGANGP(gen, disc, log_dir=log_dir, d_steps_per_g_step=1)

    initial_epoch = 0
    checkpoint_filepath = os.path.join(log_dir, "model_{epoch}.h5")

    gan.fit(
        x=dataset,
        y=None,
        epochs=50,
        initial_epoch=initial_epoch,
        # steps_per_epoch=60_000 // batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100),
            # utils.GenerateSampleGridFigureCallback(log_dir=log_dir, period=60_000 / 2 // batch_size),
            # AdaptiveBlurController(max_value=maximum_reasonable_std(image_resolution=28)),
            BlurScheduleController(schedule_type="exponential_decay", training_n_steps=50*60_000, max_value=maximum_reasonable_std(image_resolution=28)),
        ],
    )

    


    samples = gan.generate_samples()
    import numpy as np
    x = np.reshape(samples[0].numpy(), [28, 28])
    print(x.shape)
    plt.imshow(x, cmap="gray")
    plt.show()
    exit()