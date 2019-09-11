import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Callable
from enum import Enum
from gaussian_blur import GaussianBlur2D

from dataclasses import dataclass

@dataclass
class HyperParams():
    d_steps_per_g_step: int = 1
    gp_coefficient: float = 10.0
    learning_rate: float = 0.001


class WGANGP(tf.keras.Model):
    """
    example of a GAN model, where gaussian blurring is applied to the inputs of the discriminator to stabilize training.
    """
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, log_dir: str, hyperparams: HyperParams, *args, **kwargs):
        """
        TODO: add arguments for the generator and discriminator constructors.
        """
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams.learning_rate)

        self.blur = GaussianBlur2D()

        self.discriminator = discriminator
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams.learning_rate)

        self.hparams: HyperParams = hyperparams
        self.gp_coefficient = self.hparams.gp_coefficient
        # number of discriminator steps per generator step.
        self.d_steps_per_g_step = self.hparams.d_steps_per_g_step
        self.batch_size = None # will be determined dynamically when trained.
        self.save_image_summaries_interval = 5
        
        # used to keep track of progress
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.n_img = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.n_batches = tf.Variable(0, dtype=tf.int64, trainable=False)
        
        # Keras metrics to be showed during training.
        self.real_scores = tf.keras.metrics.Mean("real_scores", dtype=tf.float32)
        self.fake_scores = tf.keras.metrics.Mean("fake_scores", dtype=tf.float32)
        self.gp_term = tf.keras.metrics.Mean("gp_term", dtype=tf.float32)
        self.gen_loss = tf.keras.metrics.Mean("gen_loss", dtype=tf.float32)
        self.disc_loss = tf.keras.metrics.Mean("disc_loss", dtype=tf.float32)
        
        self.std_metric = tf.keras.metrics.Mean("std", dtype=tf.float32)

    @property
    def std(self):
        return self.blur.std

    def latents_batch(self):
        assert self.batch_size is not None
        return tf.random.uniform([self.batch_size, self.generator.latent_size])

    def generate_samples(self, latents=None, training=False):
        if latents is None:
            latents = self.latents_batch()
        return self.generator(latents, training=training)

    @tf.function()
    def gradient_penalty(self, reals, fakes):
        batch_size = reals.shape[0]
        a = tf.random.uniform([batch_size, 1, 1, 1])
        x_hat = a * reals + (1-a) * fakes

        with tf.GradientTape() as tape:
            tape.watch(x_hat)
            y_hat = self.discriminator(x_hat, training=False)

        grad = tape.gradient(y_hat, x_hat)
        norm = tf.norm(tf.reshape(grad, [batch_size, -1]), axis=1)
        return tf.reduce_mean((norm - 1.)**2)

    # @tf.function
    def discriminator_step(self, reals):
        with tf.GradientTape() as disc_tape:
            fakes = self.generate_samples(training=False)
            
            blurred_fakes = self.blur(fakes)
            blurred_reals = self.blur(reals)

            fake_scores = self.discriminator(blurred_fakes, training=True)
            real_scores = self.discriminator(blurred_reals, training=True)

            gp_term = self.gradient_penalty(blurred_reals, blurred_fakes)
            disc_loss = tf.reduce_mean(fake_scores - real_scores) + self.gp_coefficient * gp_term

            # We use the same norm term from the ProGAN authors.
            # norm_term = (tf.norm(fake_scores, axis=-1) + tf.norm(real_scores, axis=-1))
            # e_drift = 1e-4
            # disc_loss += e_drift * norm_term
        
        discriminator_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))
       
        # save metrics.
        self.fake_scores(fake_scores)
        self.real_scores(real_scores)
        self.gp_term(gp_term)
        self.disc_loss(disc_loss)
        self.std_metric(self.std)

        # images to be added as a summary
        images = (fakes, reals, blurred_fakes, blurred_reals)
        return disc_loss, images

    # @tf.function
    def generator_step(self):
        with tf.GradientTape() as gen_tape:
            fakes = self.generate_samples(training=True)
            # blurred_fakes = fakes
            blurred_fakes = self.blur(fakes)
            fake_scores = self.discriminator(blurred_fakes, training=False)
            gen_loss = - tf.reduce_mean(fake_scores)

        # save metrics.
        self.fake_scores(fake_scores)
        self.gen_loss(gen_loss)

        generator_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        return gen_loss

    # @tf.function
    def train_on_batch(self, reals, *args, **kwargs):
        self.reset_metrics()
        self.batch_size = reals.shape[0]
        with self.summary_writer.as_default():
                
            disc_loss, images = self.discriminator_step(reals)
            
            if tf.equal((self.n_batches % self.d_steps_per_g_step), 0):
                self.generator_step()
            
            if tf.equal(self.n_batches % self.save_image_summaries_interval, 0):
                self.log_image_summaries(images)

            batch_size = reals.shape[0]
            self.n_img.assign_add(batch_size)
            self.n_batches.assign_add(1)

        return [metric.result() for metric in self.metrics]

    def log_image_summaries(self, images):
        with self.summary_writer.as_default():
            fakes, reals, blurred_fakes, blurred_reals = images
                # TODO: figure out how to use image summaries properly.
            tf.summary.image("fakes", fakes, step=self.n_img)
            tf.summary.image("reals", reals, step=self.n_img)
            tf.summary.image("blurred_fakes", blurred_fakes, step=self.n_img)
            tf.summary.image("blurred_reals", blurred_reals, step=self.n_img)
            self.summary_writer.flush()


class BlurredGAN(WGANGP):
    """
    TODO: idea.
    """
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, *args, **kwargs):
        self.blur = GaussianBlur2D()        
        discriminator_with_blur = tf.keras.Sequential([
            self.blur,
            discriminator,
        ])
        super().__init__(generator, discriminator_with_blur, *args, **kwargs)


class BlurScheduleController(tf.keras.callbacks.Callback):
    def __init__(self, schedule_type: str, training_n_steps: int,  max_value: float, min_value=0.01):
        self.initial_std = max_value

        # if schedule_type == "exponential_decay":
        self.schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_std,
            decay_steps=training_n_steps / 10,
            decay_rate=0.96,
            staircase=False,
        )
        # elif schedule_type == (...)

    def on_batch_begin(self, batch, logs):
        value = self.schedule(self.model.n_img)
        self.model.std.assign(value)


class AdaptiveBlurController(tf.keras.callbacks.Callback):
    """
    Controller which adaptively reduces the amount of blurring used during
    training. To be used with the `BlurredGAN` keras model.
    
    Once the standard deviation reaches a value equal to `min_value`, the
    training stops.
    """
    def __init__(self, p=0.9, warmup_n_batches=100, min_value=0.01, max_value=23.5):
        super().__init__()
        self.p = p
        self.warmup_n_batches = warmup_n_batches
        # start with a very negative initial value.
        self.value = 0

        self.smoothing = 0.95

        # TODO: Fix this bug. self.model is None.
        self.std = max_value
        self.min_value = min_value

        self._value = None

    def on_train_begin(self, logs=None):
        self.model.std.assign(self.std)

    def on_batch_end(self, batch, logs):
        # to be used with a BlurredGAN model.
        # assert isinstance(self.model, BlurredGAN), self.model
        if batch < self.warmup_n_batches:
            return

        fake_scores = logs["fake_scores"]
        real_scores = logs["real_scores"]
        ratio = fake_scores / (real_scores + fake_scores)
        tf.summary.scalar("ratio", ratio)
        self.value = self.p * self.value + (1 - self.p) * ratio

        if 0.48 <= self.value <= 0.52 and batch > self.warmup_n_batches:
            # print("\nProblem is too easy. reducing the blur std:", self.blur_std, self.value)
            self.std = self.smoothing * self.std
            self.model.std.assign(self.std)

        if self.std < self.min_value:
            print("Reached the minimum STD. Training is complete.")
            self.model.stop_training = True
