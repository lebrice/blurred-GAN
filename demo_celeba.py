import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import blurred_gan
from blurred_gan import WGANGP, TrainingConfig, BlurredWGANGP
import callbacks

from tensorboard.plugins.hparams import api as hp

import utils
import dataclasses

def make_dataset(shuffle_buffer_size=100) -> tf.data.Dataset:
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

    data_dir = os.environ.get("DATASETS_DIR", "/tmp/datasets")
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

    import argparse
    parser = argparse.ArgumentParser()

    BlurredWGANGP.HyperParameters.add_arguments(parser)
    TrainingConfig.add_arguments(parser)

    args = parser.parse_args()
    
    hyperparameters = BlurredWGANGP.HyperParameters.from_args(args)
    config = TrainingConfig.from_args(args)
    
    print(hyperparameters)
    print(config)

    # TODO: Make multi-GPU training work
    # distribution_strategy = tf.distribute.MirroredStrategy()
    distribution_strategy = tf.distribute.get_strategy()
    num_gpus = distribution_strategy.num_replicas_in_sync
    print("Num gpus:", num_gpus)

    # Compute global batch size using number of replicas.
    global_batch_size = batch_size_per_gpu * num_gpus
    dataset = make_dataset().batch(global_batch_size)

    total_n_examples = 202_599
    steps_per_epoch = total_n_examples // global_batch_size

    results_dir = "results"
    config.log_dir = utils.create_result_subdir(results_dir, "celeba")
    config.checkpoint_dir = config.log_dir + "/checkpoints"    

    gen = DCGANGenerator()
    disc = DCGANDiscriminator()
    gan = blurred_gan.BlurredWGANGP(gen, disc, hyperparams=hyperparameters, config=config)
    
    checkpoint = tf.train.Checkpoint(gan=gan)    
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=config.checkpoint_dir,
        max_to_keep=5,
        keep_checkpoint_every_n_hours=1
    )

    hparams_file_path = os.path.join(config.log_dir, "hyper_parameters.json")
    train_config_file_path = os.path.join(config.log_dir, "train_config.json")

    if manager.latest_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        status.assert_existing_objects_matched()
        gan.hparams = BlurredWGANGP.HyperParameters.from_json(hparams_file_path)
        gan.config = TrainingConfig.from_json(train_config_file_path)
        print("Loaded model weights from previous checkpoint:", checkpoint)
        print(f"Model was previously trained on {gan.n_img.numpy()} images")
        tf.summary.experimental.set_step(gan.n_img)
    
    print("Hparams:", gan.hparams)
    print("Train config:", gan.config)

    gan.hparams.save_json(hparams_file_path)
    gan.config.save_json(train_config_file_path)

    # manager.save()

    metric_callbacks = [
        callbacks.FIDMetricCallback(
            image_preprocessing_fn=lambda img: tf.image.resize(img, [299, 299]),
            num_samples=100,
            every_n_examples=50_000,
        ),
        callbacks.SWDMetricCallback(
            image_preprocessing_fn=lambda img: utils.NHWC_to_NCHW(tf.convert_to_tensor(img)),
            num_samples=1000,
            every_n_examples=50_000,
        ),
    ]

    try:
        gan.fit(
            x=dataset,
            y=None,
            epochs=epochs,
            initial_epoch=gan.n_img // total_n_examples,
            callbacks=[
                # log the hyperparameters used for this run
                hp.KerasCallback(config.log_dir, hyperparameters.asdict()),
                # generate a grid of samples
                callbacks.GenerateSampleGridCallback(log_dir=config.log_dir, every_n_examples=5_000),
                # # FIXME: these controllers need to be cleaned up a tiny bit.
                # AdaptiveBlurController(max_value=hyperparameters.initial_blur_std),
                callbacks.BlurDecayController(total_n_training_examples=total_n_examples * epochs, max_value=5),
                
                # heavy metric callbacks
                *metric_callbacks,
                callbacks.SaveModelCallback(manager, n=10_000),
                callbacks.LogMetricsCallback()
            ]
        )
    except KeyboardInterrupt:
        manager.save()
    # Save the model
    print("Done training.")


    samples = gan.generate_samples()
    import numpy as np
    x = np.reshape(samples[0].numpy(), [28, 28])
    print(x.shape)
    plt.imshow(x, cmap="gray")
    plt.show()
    exit()