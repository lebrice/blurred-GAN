import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Callable

from gaussian_blur import GaussianBlur2D


class BlurredGAN(tf.keras.Model):
    """
    Simple GAN model, which applies gaussian blur to the inputs of the discriminator.
    """
    def __init__(self, generator, discriminator, d_steps_per_g_step=1, *args, **kwargs):
        """
        TODO: add arguments for the generator and discriminator constructors.
        """
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam()

        self.discriminator = discriminator
        self.discriminator_optimizer = tf.keras.optimizers.Adam()

        self.blur = GaussianBlur2D()

        self.gp_coefficient = 10.0
        self.batch_size = None

        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)
        
        # Keras metrics to be showed during training.
        self.real_scores = tf.keras.metrics.Mean("real_scores", dtype=tf.float32)
        self.fake_scores = tf.keras.metrics.Mean("fake_scores", dtype=tf.float32)
        self.gp_term = tf.keras.metrics.Mean("gp_term", dtype=tf.float32)
        self.gen_loss = tf.keras.metrics.Mean("gen_loss", dtype=tf.float32)
        self.disc_loss = tf.keras.metrics.Mean("disc_loss", dtype=tf.float32)
        self.std_metric = tf.keras.metrics.Mean("std", dtype=tf.float32)

        # number of discriminator steps per generator step.
        self.d_steps_per_g_step = d_steps_per_g_step

    @property
    def std(self):
        return self.blur.std

    @std.setter
    def std(self, value):
        self.blur.std = value

    def latents_batch(self):
        assert self.batch_size is not None
        return tf.random.uniform([self.batch_size, self.generator.latent_size])

    def generate_samples(self, latents=None, training=False):
        if latents is None:
            latents = self.latents_batch()
        return self.generator(latents, training=training)

    def gradient_penalty(self, reals, fakes):
        # interpolate between real and fakes
        batch_size = reals.shape[0]
        a = tf.random.uniform([batch_size, 1, 1, 1])
        x_hat = reals + a * (fakes - reals)

        with tf.GradientTape() as tape:
            tape.watch(x_hat)    
            y_hat = self.discriminator(x_hat, training=True)

        grad = tape.gradient(y_hat, x_hat)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp_term = self.gp_coefficient * tf.reduce_mean((norm - 1.)**2)
        return gp_term

    @tf.function
    def discriminator_step(self, reals):
        self.batch_size = reals.shape[0]
        with tf.GradientTape() as disc_tape:
            fakes = self.generate_samples(training=False)

            blurred_fakes = self.blur(fakes)
            blurred_reals = self.blur(reals)

            fake_scores = self.discriminator(blurred_fakes, training=True)
            real_scores = self.discriminator(blurred_reals, training=True)

            gp_term = self.gradient_penalty(blurred_reals, blurred_fakes)
            disc_loss = (fake_scores - real_scores) + gp_term

            norm_term = (tf.norm(fake_scores, axis=-1) + tf.norm(real_scores, axis=-1))
            e_drift = 1e-4
            disc_loss += 1e-3 * norm_term

        # save metrics.
        self.fake_scores(fake_scores)
        self.real_scores(real_scores)
        self.gp_term(gp_term)
        self.disc_loss(disc_loss)
        self.std_metric(self.std)

        # with tf.device("cpu"):
        #     # TODO: figure out how to use image summaries.
            # tf.summary.image("fakes", fakes, step=self.step.numpy())
            # tf.summary.image("reals", reals, step=self.step.numpy())

        discriminator_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))

        return disc_loss

    @tf.function
    def generator_step(self):
        with tf.GradientTape() as gen_tape:
            fakes = self.generate_samples(training=True)
            fake_scores = self.discriminator(fakes, training=False)
            gen_loss = tf.reduce_mean(fake_scores)

        # save metrics.
        self.fake_scores(fake_scores)
        self.gen_loss(gen_loss)

        generator_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        return gen_loss

    def train_on_batch(self, x, *args, **kwargs):
        batch_size = x.shape[0]
        reals = x
        self.discriminator_step(reals)
        self.step.assign_add(batch_size)
        if tf.equal((self.step / batch_size) % self.d_steps_per_g_step, 0):
            self.generator_step()
        return [metric.result() for metric in self.metrics]


class AdaptiveBlurController(tf.keras.callbacks.Callback):
    """
    Controller which adaptively reduces the amount of blurring used during
    training. To be used with the `BlurredGAN` keras model.

    An exponential moving average (with coefficient `p`) of the scores given
    to fake samples by the discriminator is kept during training.
    while this moving average falls within the interval [`threshold`, 0], the
    standard deviation is reduced by a factor of `smoothing` (defaults to 0.95)
    
    Once the standard deviation reaches a value equal to `min_value`, the
    training stops.
    """
    def __init__(self, threshold=-1.0, p=0.9, smoothing=0.95, max_value=23.8, min_value=0.01):
        super().__init__()

        self.threshold = threshold
        self.smoothing = smoothing
        self.p = p

        # start with a very negative initial value.
        self.value = 10 * self.threshold

        # TODO: Fix this bug. self.model is None.
        self.std = max_value
        self.min_value = min_value

    def on_batch_end(self, batch, logs):
        # to be used with a BlurredGAN model.
        # assert isinstance(self.model, BlurredGAN), self.model
        new_value = logs["fake_scores"]
        self.value = self.p * self.value + (1 - self.p) * new_value
        if self.threshold <= self.value <= 0:
            # print("\nProblem is too easy. reducing the blur std:", self.blur_std, self.value)
            self.std = self.smoothing * self.std
            self.model.std = self.std

        if self.std < self.min_value:
            print("Reached the minimum STD. Training is complete.")
            self.model.stop_training = True

        tf.summary.scalar("blur_std", self.std)
