"""
Wasserstein GAN in Tensorflow 2.0
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import *
from enum import Enum
import dataclasses
from dataclasses import dataclass, InitVar, field
import os
import json
from contextlib import contextmanager

from gaussian_blur import GaussianBlur2D
from simple_parsing import ParseableFromCommandLine
from utils import JsonSerializable


@dataclass
class TrainingConfig(JsonSerializable, ParseableFromCommandLine):
    log_dir: str = "results/log"
    checkpoint_dir: str = "results/log/checkpoints"
    save_image_summaries_interval: int = 50


class WGAN(tf.keras.Model):
    """
    Wasserstein GAN
    """


    @dataclass
    class HyperParameters(JsonSerializable, ParseableFromCommandLine):
        """
        Dataclass containing the hyperparameters of the Model
        """
        learning_rate: float = 0.001
        d_steps_per_g_step: int = 1
        batch_size: int = 32
        global_batch_size: int = 32
        optimizer: str = "adam"

    
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, hyperparams: HyperParameters, config: TrainingConfig, *args, **kwargs):
        """
        Creates the GAN, using the given `generator` and `discriminator` models.
        """
        super().__init__(*args, **kwargs)
        # hyperparameters
        self.hparams: WGAN.HyperParameters = hyperparams
        

        self.generator = generator 
        self.generator.optimizer = tf.keras.optimizers.get(self.hparams.optimizer)
        self.generator.optimizer.learning_rate = self.hparams.learning_rate

        self.discriminator = discriminator
        self.discriminator.optimizer = tf.keras.optimizers.get(self.hparams.optimizer)
        self.discriminator.optimizer.learning_rate = self.hparams.learning_rate

        # number of discriminator steps per generator step.
        self.d_steps_per_g_step = self.hparams.d_steps_per_g_step
        self.batch_size = None # will be determined dynamically when trained.
        
        self.config = config
        self.summary_writer = tf.summary.create_file_writer(config.log_dir)
        # used to keep track of progress
        self.n_img = tf.Variable(0, dtype=tf.int64, trainable=False, name="n_img")
        self.n_batches = tf.Variable(0, dtype=tf.int64, trainable=False, name="n_batches")
        
        # Keras metrics to be showed during training.
        self.real_scores_metric = tf.keras.metrics.Mean("real_scores", dtype=tf.float32)
        self.fake_scores_metric = tf.keras.metrics.Mean("fake_scores", dtype=tf.float32)
        self.gen_loss_metric = tf.keras.metrics.Mean("gen_loss", dtype=tf.float32)
        self.disc_loss_metric = tf.keras.metrics.Mean("disc_loss", dtype=tf.float32)
        
        # BUG: for some reason, a model needs a non-None value for the 'optimizer' attribute before it can be trained with the .fit method.
        self.optimizer = "unused"

        self.strategy: tf.distribute.DistributionStrategy = tf.distribute.get_strategy()
    
    def latents_batch(self):
        assert self.batch_size is not None
        return tf.random.uniform([self.batch_size, self.generator.input_shape[-1]])

    def generate_samples(self, latents=None, training=False):
        if latents is None:
            latents = self.latents_batch()
        return self.generator(latents, training=training)

    @tf.function
    def discriminator_loss(self, reals, fakes, real_scores, fake_scores):
        return tf.reduce_sum(fake_scores - real_scores) * (1. / self.hparams.global_batch_size)

    @tf.function
    def discriminator_step(self, reals):
        with tf.GradientTape() as disc_tape:
            fakes = self.generate_samples(training=False)
            fake_scores = self.discriminator(fakes, training=True)
            real_scores = self.discriminator(reals, training=True)
            disc_loss = self.discriminator_loss(reals, fakes, real_scores, fake_scores)
        
        discriminator_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))
        # save metrics.
        # self.fake_scores_metric(fake_scores)
        # self.real_scores_metric(real_scores)
        # self.disc_loss_metric(disc_loss)

        # images to be added as a summary
        # BUG: Currently, it seems like we can't have image summaries inside a tf.function (graph). (not thoroughly tested this yet.)
        # hence we pass the images outside of this 'graphified' function.
        images = (fakes, reals)
        return disc_loss, images

    @tf.function
    def generator_loss(self, fake_scores):
        return - tf.reduce_sum(fake_scores) * (1. / self.hparams.global_batch_size)

    @tf.function
    def generator_step(self):
        with tf.GradientTape() as gen_tape:
            fakes = self.generate_samples(training=True)
            fake_scores = self.discriminator(fakes, training=False)
            gen_loss = self.generator_loss(fake_scores)
        
        generator_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        
        # save metrics.
        # self.fake_scores_metric(fake_scores)
        # self.gen_loss_metric(gen_loss)
        return gen_loss

    # @tf.function
    def train_on_batch(self, reals, *args, **kwargs):
        self.reset_metrics()
        # NOTE: by default keras resets the metrics only after each epoch:
        # if kwargs.get("reset_metrics"):
        #     print("resetting metrics")
        #     self.reset_metrics()

        self.batch_size = reals.shape[0]
        tf.summary.experimental.set_step(self.n_img)

        disc_loss, images = self.discriminator_step(reals)
        self.images = images
        
        if tf.equal(self.n_batches % self.d_steps_per_g_step, 0):
            self.generator_step()
        
        self.log_image_summaries()

        batch_size = reals.shape[0]
        self.n_img.assign_add(batch_size)
        self.n_batches.assign_add(1)

        #BUG: the order of metrics in the `metrics` attribute is not the same as in the `metrics_names` attribute.
        # the first value in the returned list should be the 'loss'
        metric_results = [0] # some dummy 'loss', which doesn't apply here.
        for metric_name in self.metrics_names[1:]:
            result = [m.result() for m in self.metrics if m.name == metric_name]
            assert len(result) == 1, "duplicate metric names"
            metric_results += result
        return metric_results

    def log_image_summaries(self):
        with self.record_image_summaries():
            fakes, reals = self.images
            tf.summary.image("fakes", fakes)
            tf.summary.image("reals", reals)
    
    def summary(self):
        print("Discriminator:")
        self.discriminator.summary()
        print("Generator:")
        self.generator.summary()
        print(f"Total params: {self.count_params():,}")

    @tf.function
    def _saving_image_summaries(self) -> bool:
        """
        Returns True if image summaries should be saved for the current batch.
        """
        return tf.equal(self.n_batches % self.config.save_image_summaries_interval, 0)

    @contextmanager
    def record_image_summaries(self):
        """
        Context manager that enables the recording of image summaries once every `self.config.save_image_summaries_interval` batches.
        """
        with tf.device("cpu"), self.summary_writer.as_default():
            with tf.summary.record_if(self._saving_image_summaries):
                yield


    def count_params(self):
        return self.discriminator.count_params() + self.generator.count_params()

    def save_weights(self, filepath, overwrite=True, save_format=None):
        self.discriminator.save_weights(filepath+"_discriminator", overwrite, save_format)
        self.generator.save_weights(filepath+"_generator", overwrite, save_format)


@tf.function()
def gradient_penalty(discriminator, reals, fakes):
    batch_size = tf.shape(reals)[0]
    a = tf.random.uniform([batch_size, 1, 1, 1])
    # x_hat = a * reals + (1-a) * fakes
    x_hat = reals + a * (fakes - reals)
    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        y_hat = discriminator(x_hat, training=False)

    grad = tape.gradient(y_hat, x_hat)
    norm = tf.norm(tf.reshape(grad, [batch_size, -1]), axis=1)
    return tf.reduce_mean((norm - 1.)**2)


class WGANGP(WGAN):
    """
    Wasserstein GAN with Gradient Penalty loss
    """


    @dataclass
    class HyperParameters(WGAN.HyperParameters):
        """Hyperparameters of a WGAN model with Gradient Penalty loss."""
        """Coefficient from the progan authors which penalizes critic outputs for having a large magnitude."""
        e_drift: float = 1e-4
        """Multiplying coefficient for the gradient penalty term of the loss equation. (10.0 is the default value, and was used by the PROGAN authors.)"""
        gp_coefficient: float = 10.0


    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, hyperparams: HyperParameters, config: TrainingConfig, *args, **kwargs):
        """
        Creates the model, using the given `generator` and `discriminator` models.
        """
        super().__init__(generator, discriminator, hyperparams, config, *args, **kwargs)
        self.gp_term_metric = tf.keras.metrics.Mean("gp_term", dtype=tf.float32)
        self.norm_term_metric = tf.keras.metrics.Mean("norm_term", dtype=tf.float32)

    @tf.function
    def discriminator_loss(self, reals, fakes, real_scores, fake_scores):
        disc_loss = super().discriminator_loss(reals, fakes, real_scores, fake_scores)
        
        # Gradient penalty addition.
        gp_term = self.hparams.gp_coefficient * gradient_penalty(self.discriminator, reals, fakes)
        self.gp_term_metric(gp_term)
        disc_loss += gp_term

        # We use the same norm term from the ProGAN authors.
        norm_term = self.hparams.e_drift * (tf.norm(fake_scores, axis=-1) + tf.norm(real_scores, axis=-1))
        self.norm_term_metric(norm_term)
        disc_loss += norm_term
        return disc_loss
