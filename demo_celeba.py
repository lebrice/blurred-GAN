import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import blurred_gan
from blurred_gan import WGANGP, HyperParams, TrainingConfig, BlurredGAN
from callbacks import AdaptiveBlurController, BlurScheduleController

import utils


def celeba_dataset(shuffle_buffer_size=100) -> tf.data.Dataset:
    """Modern Tensorflow input pipeline for the CelebA dataset"""

    @tf.function
    def take_image(example):
        return example["image"]

    @tf.function
    def normalize(image):
        return (tf.cast(image, tf.float32) - 127.5) / 127.5

    @tf.function
    def resize_image(image):
        image = tf.image.resize(image, [128, 128])
        return image

    @tf.function
    def preprocess_images(image):
        image = normalize(image)
        image = resize_image(image)
        return image
    import os
    from os import environ
    data_dir = environ.get("DATASETS_DIR", "/tmp/datasets")
    celeba_dataset = tfds.load(name="celeb_a", data_dir=data_dir, split=tfds.Split.ALL)
    celeba = (celeba_dataset
              .map(take_image)
              .batch(16)  # make preprocessing faster by batching inputs.
              .map(preprocess_images)
              .unbatch()
            #   .cache("./cache")
              .shuffle(shuffle_buffer_size)
              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
              )
    return celeba


class DCGANGenerator(tf.keras.Sequential):
    def __init__(self, latent_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_size = latent_size
        self.add(layers.Dense(4*4*512, use_bias=False, input_shape=(self.latent_size,)))
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Reshape((4, 4, 512)))
        assert self.output_shape == (None, 4, 4, 512) # Note: None is the batch size

        self.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.output_shape == (None, 4, 4, 512), self.output_shape
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.output_shape == (None, 8, 8, 256), self.output_shape
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.output_shape == (None, 16, 16, 128), self.output_shape
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.output_shape == (None, 32, 32, 64), self.output_shape
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.output_shape == (None, 64, 64, 32), self.output_shape
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.output_shape == (None, 128, 128, 16), self.output_shape
        self.add(layers.BatchNormalization())
        self.add(layers.LeakyReLU())

        self.add(layers.Conv2D(3, (5, 5), padding='same', use_bias=False, activation='tanh'))
        assert self.output_shape == (None, 128, 128, 3), self.output_shape


class DCGANDiscriminator(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add(layers.Conv2D(16, 5, strides=2, padding='same', input_shape=[128, 128, 3]))
        self.add(layers.LeakyReLU())
        self.add(layers.Dropout(0.3))

        self.add(layers.Conv2D(32, 5, strides=2, padding='same'))
        self.add(layers.LeakyReLU())
        self.add(layers.Dropout(0.3))

        self.add(layers.Conv2D(64, 5, strides=2, padding='same'))
        self.add(layers.LeakyReLU())
        self.add(layers.Dropout(0.3))

        self.add(layers.Conv2D(128, 5, strides=2, padding='same'))
        self.add(layers.LeakyReLU())
        self.add(layers.Dropout(0.3))

        self.add(layers.Conv2D(256, 5, strides=2, padding='same'))
        self.add(layers.LeakyReLU())
        self.add(layers.Dropout(0.3))

        self.add(layers.Conv2D(512, 5, strides=2, padding='same'))
        self.add(layers.LeakyReLU())
        self.add(layers.Dropout(0.3))

        self.add(layers.Flatten())
        self.add(layers.Dense(1, activation="linear"))


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import datetime

    tf.random.set_seed(123123)

    epochs = 10
    batch_size_per_gpu = 32
    
    log_dir = utils.create_result_subdir("results", "celeba")
    checkpoint_filepath = log_dir + '/model_{epoch}.h5'

    train_config = TrainingConfig(
        log_dir=log_dir,
    )
    
    # strategy = tf.distribute.CentralStorageStrategy()
    num_gpus = 1 #strategy.num_replicas_in_sync
    print("Num gpus:", num_gpus)

    # Compute global batch size using number of replicas.
    global_batch_size = batch_size_per_gpu * num_gpus
    dataset = celeba_dataset().batch(global_batch_size)

    steps_per_epoch = 202_599 // global_batch_size

    # with strategy.scope():
    gen = DCGANGenerator()
    disc = DCGANDiscriminator()

    hyperparameters = HyperParams(
        d_steps_per_g_step=1,
        gp_coefficient=10.0,
        learning_rate=0.001,
        initial_blur_std=23.5,
    )

    

    gan = WGANGP(gen, disc, hyperparams=hyperparameters, config=train_config)
    gan.summary()
    gan.fit(
        x=dataset,
        y=None,
        epochs=epochs,
        steps_per_epoch=202_599 // global_batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_freq=100_000),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100, profile_batch=0), # BUG: profile_batch=0 was put there to fix Tensorboard not updating correctly. 
            utils.GenerateSampleGridFigureCallback(log_dir=log_dir, period=100),
            AdaptiveBlurController(), # FIXME: this controller is not yet operational.
            BlurScheduleController(total_n_training_batches=steps_per_epoch * epochs)
        ]
    )