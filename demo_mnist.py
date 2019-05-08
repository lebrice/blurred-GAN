import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import blurred_gan
from blurred_gan import BlurredGAN, AdaptiveBlurController

import utils


def dataset(shuffle_buffer_size=100) -> tf.data.Dataset:
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

    # (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    # dataset = tf.data.Dataset.from_tensor_slices(train_images)

    # stats = lambda x: tf.print(tf.reduce_min(x), tf.reduce_max(x), tf.shape(x))

    # stats(next(iter(dataset.take(1))))
    # stats(next(iter(dataset.map(normalize).take(1))))

    dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)

    # stats(next(iter(dataset.map(take_image).take(1))))
    # stats(next(iter(dataset.map(take_image).map(preprocess_images).take(1))))

    # exit()

    dataset = (dataset
        .map(take_image)
        .batch(16)  # make preprocessing faster by batching inputs.
        .map(preprocess_images)
        .apply(tf.data.experimental.unbatch())
    #   .cache("./cache")
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

        self.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
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
    batch_size = 16

    gen = Generator()
    disc = Discriminator()

    gan = BlurredGAN(gen, disc, d_steps_per_g_step=1)

    dataset = dataset().batch(batch_size)

    results_dir = utils.create_result_subdir("results", "mnist")
    checkpoint_dir = results_dir

    checkpoint_filepath = checkpoint_dir + '/self_{epoch}.h5'

    # we add a useless 'label' for the fit method to work.
    dataset = dataset.map(lambda v: (v, 0))


    gan.fit(
        x=dataset,
        y=None,
        epochs=20,
        # steps_per_epoch=50_000 // batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath),
            tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir, update_freq=100),
            # utils.GenerateSampleGridFigureCallback(log_dir=checkpoint_dir, period=100),
            AdaptiveBlurController(),
        ]
    )
