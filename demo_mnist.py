import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import blurred_gan
from blurred_gan import WGANGP, TrainingConfig, HyperParams
from callbacks import AdaptiveBlurController, BlurScheduleController

from tensorboard.plugins.hparams import api as hp

import utils
import dataclasses


def make_dataset(shuffle_buffer_size=256) -> tf.data.Dataset:
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


class DCGANGenerator(tf.keras.Sequential):
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


class DCGANDiscriminator(tf.keras.Sequential):
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

    tf.random.set_seed(123123)

    epochs = 10
    batch_size_per_gpu = 32
    
    log_dir = utils.create_result_subdir("results", "mnist")
    checkpoint_filepath = log_dir + '/model_{epoch}.h5'

    train_config = TrainingConfig(
        log_dir=log_dir,
    )
    
    # strategy = tf.distribute.CentralStorageStrategy()
    num_gpus = 1 #strategy.num_replicas_in_sync
    print("Num gpus:", num_gpus)

    # Compute global batch size using number of replicas.
    global_batch_size = batch_size_per_gpu * num_gpus
    dataset = make_dataset().batch(global_batch_size)

    total_n_examples = 60_000
    steps_per_epoch = total_n_examples // global_batch_size

    # with strategy.scope():
    gen = DCGANGenerator()
    disc = DCGANDiscriminator()

    hyperparameters = HyperParams(
        d_steps_per_g_step=1,
        gp_coefficient=10.0,
        learning_rate=0.001,
        # initial_blur_std=,
    )
    gan = WGANGP(gen, disc, hyperparams=hyperparameters, config=train_config)
    gan.summary()
    gan.fit(
        x=dataset.take(10),
        y=None,
        epochs=1, #epochs,
        # steps_per_epoch=steps_per_epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_freq='epoch'),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100, profile_batch=0), # BUG: profile_batch=0 was put there to fix Tensorboard not updating correctly. 
            utils.GenerateSampleGridFigureCallback(log_dir=log_dir, period=100),
            AdaptiveBlurController(max_value=hyperparameters.initial_blur_std), # FIXME: this controller is not yet operational.
            BlurScheduleController(total_n_training_batches=steps_per_epoch * epochs, max_value=hyperparameters.initial_blur_std),

            hp.KerasCallback(log_dir, hyperparameters.asdict())
        ]
    )

    import metrics
      
    def fid_score(n=1000) -> float:
        import tensorflow_hub as hub
        model_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

        model = hub.KerasLayer(model_url, output_shape=[2048], input_shape=[299,299,3], trainable=False)
        image_preprocessing_layer = tf.keras.layers.Lambda(lambda img: tf.image.grayscale_to_rgb(tf.image.resize(img, [299,299])))
        feature_extractor = tf.keras.Sequential([
            image_preprocessing_layer,
            model,
        ])
        
        reals = make_dataset().take(n)
        fakes = gan.generator.predict(tf.random.uniform((n, gan.generator.input_shape[-1])))
        
        return metrics.evaluate_fid(reals, fakes, feature_extractor)
    
    import time
    for n in (10, 50, 100, 200, 500, 1000):
        start = time.time()
        fid = fid_score(n)
        length = time.time() - start
        print(f"n: {n}, time: {length} secs, ratio: {length / n} fid: {fid}")
    # samples = gan.generate_samples()
    # import numpy as np
    # x = np.reshape(samples[0].numpy(), [28, 28])
    # print(x.shape)
    # plt.imshow(x, cmap="gray")
    # plt.show()
    # exit()