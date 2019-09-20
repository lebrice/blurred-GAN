import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Callable
from enum import Enum
import dataclasses
from dataclasses import dataclass

from gaussian_blur import GaussianBlur2D

@dataclass
class HyperParams():
    d_steps_per_g_step: int = 1
    gp_coefficient: float = 10.0
    learning_rate: float = 0.001
    
    initial_blur_std: float = 23.5

    """Coefficient from the progan authors which penalizes critic outputs for having a large magnitude."""
    e_drift: float = 1e-4

    def asdict(self):
        return dataclasses.asdict(self)

@dataclass
class TrainingConfig():
    log_dir: str
    save_image_summaries_interval: int = 50


@tf.function()
def gradient_penalty(discriminator, reals, fakes):
    batch_size = reals.shape[0]
    a = tf.random.uniform([batch_size, 1, 1, 1])
    x_hat = a * reals + (1-a) * fakes

    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        y_hat = discriminator(x_hat, training=False)

    grad = tape.gradient(y_hat, x_hat)
    norm = tf.norm(tf.reshape(grad, [batch_size, -1]), axis=1)
    return tf.reduce_mean((norm - 1.)**2)

class WGANGP(tf.keras.Model):
    """
    Wasserstein GAN with Gradient Penalty.
   
    """
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, hyperparams: HyperParams, config: TrainingConfig, *args, **kwargs):
        """
        Creates the GAN, using the given `generator` and `discriminator` models.
        """
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams.learning_rate)

        self.discriminator = discriminator
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams.learning_rate)

        self.blur = GaussianBlur2D(initial_std=hyperparams.initial_blur_std)

        # hyperparameters
        self.hparams: HyperParams = hyperparams
        self.gp_coefficient = self.hparams.gp_coefficient
        # number of discriminator steps per generator step.
        self.d_steps_per_g_step = self.hparams.d_steps_per_g_step
        self.batch_size = None # will be determined dynamically when trained.
        
        self.config = config
        self.log_dir = config.log_dir
        self.summary_writer = tf.summary.create_file_writer(config.log_dir)
        # used to keep track of progress
        self.n_img = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.n_batches = tf.Variable(0, dtype=tf.int64, trainable=False)
        
        # Keras metrics to be showed during training.
        self.real_scores = tf.keras.metrics.Mean("real_scores", dtype=tf.float32)
        self.fake_scores = tf.keras.metrics.Mean("fake_scores", dtype=tf.float32)
        self.gp_term = tf.keras.metrics.Mean("gp_term", dtype=tf.float32)
        self.gen_loss = tf.keras.metrics.Mean("gen_loss", dtype=tf.float32)
        self.disc_loss = tf.keras.metrics.Mean("disc_loss", dtype=tf.float32)
        
        #TODO: remove later
        self.std_metric = tf.keras.metrics.Mean("std", dtype=tf.float32)
        
        # BUG: for some reason, a model needs a non-None value for the 'optimizer' attribute before it can be trained with the .fit method.
        self.optimizer = "unused"

    
    def latents_batch(self):
        assert self.batch_size is not None
        return tf.random.uniform([self.batch_size, self.generator.input_shape[-1]])

    def generate_samples(self, latents=None, training=False):
        if latents is None:
            latents = self.latents_batch()
        return self.generator(latents, training=training)

    @tf.function
    def discriminator_step(self, reals):
        with tf.GradientTape() as disc_tape:
            fakes = self.generate_samples(training=False)

            # blurred_reals = reals
            # blurred_fakes = fakes
            blurred_reals = self.blur(reals)
            blurred_fakes = self.blur(fakes)

            fake_scores = self.discriminator(blurred_fakes, training=True)
            real_scores = self.discriminator(blurred_reals, training=True)
            disc_loss = tf.reduce_mean(fake_scores - real_scores)

            # Gradient penalty addition.
            gp_term = gradient_penalty(self.discriminator, reals, fakes)
            disc_loss += self.gp_coefficient * gp_term

            # We use the same norm term from the ProGAN authors.
            norm_term = (tf.norm(fake_scores, axis=-1) + tf.norm(real_scores, axis=-1))
            disc_loss += self.hparams.e_drift * norm_term
        
        discriminator_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))
       
        # save metrics.
        self.fake_scores(fake_scores)
        self.real_scores(real_scores)
        self.gp_term(gp_term)
        self.disc_loss(disc_loss)

        # TODO: remove later
        self.std_metric(self.std)

        # images to be added as a summary
        # BUG: Currently, it seems like we can't have image summaries inside a tf.function (graph). (not thoroughly tested this yet.)
        # hence we pass the images outside of this 'graphified' function.
        images = (fakes, reals, blurred_fakes, blurred_reals)
        return disc_loss, images

    @tf.function
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
        # if kwargs.get("reset_metrics"):
        #     print("resetting metrics")
        #     self.reset_metrics()

        self.batch_size = reals.shape[0]
        tf.summary.experimental.set_step(self.n_img)

        disc_loss, images = self.discriminator_step(reals)
        self.images = images

        if tf.equal((self.n_batches % self.d_steps_per_g_step), 0):
            self.generator_step()
        
        if tf.equal(self.n_batches % self.config.save_image_summaries_interval, 0):
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
        with tf.device("cpu"), self.summary_writer.as_default():
            fakes, reals, blurred_fakes, blurred_reals = self.images
            tf.summary.image("fakes", fakes)
            tf.summary.image("reals", reals)
            tf.summary.image("blurred_fakes", blurred_fakes)
            tf.summary.image("blurred_reals", blurred_reals)
            # self.summary_writer.flush()
    
    def summary(self):
        print("Discriminator:")
        self.discriminator.summary()
        print("Generator:")
        self.generator.summary()
        print(f"Total params: {self.count_params():,}")

    def count_params(self):
        return self.discriminator.count_params() + self.generator.count_params()

    @property
    def std(self):
        return self.blur.std

class BlurredGAN(WGANGP):
    """
    IDEA: Simple variation on the WGAN-GP (or any GAN architecture, for that matter) where we     
    """
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, *args, **kwargs):
        blur = GaussianBlur2D(initial_std=23.5, input_shape=discriminator.input_shape[1:])
        discriminator_with_blur = tf.keras.Sequential([
            # tf.keras.layers.InputLayer(input_shape=discriminator.input_shape[1:]),
            blur,
            discriminator,
        ])
        super().__init__(generator, discriminator_with_blur, *args, **kwargs)
        self.blur = blur
        self.std_metric = tf.keras.metrics.Mean("std", dtype=tf.float32)

    @property
    def std(self):
        return self.blur.std
    
    def discriminator_step(self, reals):
        with self.summary_writer.as_default():
            disc_loss, images = super().discriminator_step(reals)
            self.std_metric(self.std) # add this 'std' metric
            return disc_loss, images
