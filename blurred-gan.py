import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import tensorflow.keras
from typing import Callable


from gaussian_blur import GaussianBlur2D, Function


class Generator(tf.keras.Sequential):
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


class Discriminator(tf.keras.Sequential):
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
        self.add(layers.Dense(1))


class DiscriminatorWithBlur(tf.keras.Model):
    def __init__(self, blur_std: float, discriminator_constructor: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blur_std = blur_std
        self.blurring_layer = GaussianBlur2D(self.blur_std)
        self.disc = discriminator_constructor()

    def call(self, x: tf.Tensor, training=False):
        self.blurring_layer.std = self.blur_std
        x = self.blurring_layer(x)
        x = self.disc(x, training=training)
        return x


class GAN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = Generator()
        self.generator_optimizer = tf.keras.optimizers.Adam()

        self.std = tf.Variable(1.0, name="blur_std", trainable=False)
        self.discriminator = DiscriminatorWithBlur(self.std, Discriminator)
        self.discriminator_optimizer = tf.keras.optimizers.Adam()

        self.gp_coefficient = 10.0
        self.batch_size = None

        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.real_scores = tf.keras.metrics.Mean("real_scores", dtype=tf.float32)
        self.fake_scores = tf.keras.metrics.Mean("fake_scores", dtype=tf.float32)
        self.gp_term = tf.keras.metrics.Mean("gp_term", dtype=tf.float32)

        self.gen_loss = tf.keras.metrics.Mean("gen_loss", dtype=tf.float32)
        self.disc_loss = tf.keras.metrics.Mean("disc_loss", dtype=tf.float32)

        self.d_steps_per_g_step = 5

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

    @Function
    def discriminator_step(self, reals):
        self.batch_size = reals.shape[0]
        with tf.GradientTape() as disc_tape:
            fakes = self.generate_samples(training=False)
            fake_scores = self.discriminator(fakes, training=True)
            real_scores = self.discriminator(reals, training=True)
            gp_term = self.gradient_penalty(reals, fakes)
            disc_loss = tf.reduce_mean(fake_scores - real_scores) + gp_term

        # save metrics.
        self.fake_scores(fake_scores)
        self.real_scores(real_scores)
        self.gp_term(gp_term)
        self.disc_loss(disc_loss)

        # TODO: figure out how to use image summaries.
        # tf.summary.image("fakes", fakes, step=self.step)
        # tf.summary.image("reals", reals, step=self.step)

        discriminator_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))

        return disc_loss

    @Function
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

    @Function
    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
        batch_size = x.shape[0]
        self.discriminator_step(x)
        self.step.assign_add(batch_size)
        if tf.equal((self.step / batch_size) % self.d_steps_per_g_step, 0):
            self.generator_step()
        return [metric.result() for metric in self.metrics]

def celeba_dataset(batch_size=32, shuffle_buffer_size=100) -> tf.data.Dataset:
    """Modern Tensorflow input pipeline for the CelebA dataset"""

    @Function
    def take_image(example):
        return example["image"]

    @Function
    def normalize(image):
        return (tf.cast(image, tf.float32) - 127.5) / 127.5

    @Function
    def resize_image(image):
        image = tf.image.resize(image, [128, 128])
        return image

    @Function
    def preprocess_images(image):
        image = normalize(image)
        image = resize_image(image)
        return image

    celeba_dataset = tfds.load(name="celeb_a", split=tfds.Split.ALL)
    celeba = (celeba_dataset
              .map(take_image)
              .batch(16)
              .map(preprocess_images)
            #   .cache("./cache")
              .apply(tf.data.experimental.unbatch())
              .shuffle(shuffle_buffer_size)
              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
              .batch(batch_size)
              )
    return celeba


class GenerateSampleGridFigureCallback(tf.keras.callbacks.Callback):
    pass  # TODO


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import datetime
    gan = GAN()

    epochs = 10
    batch_size = 16
    d_steps_per_g_step = 3

    dataset = celeba_dataset(batch_size=batch_size)

    results_dir = "./results"
    checkpoint_dir = results_dir + "/funfun"
    checkpoint_filepath = checkpoint_dir + '/model_{epoch}.h5'
    
    dataset = dataset.map(lambda v: (v, 0))

    gan.fit(
        x=dataset,
        y=None,
        epochs=1,
        steps_per_epoch=202_599 // batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath),
            tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir, update_freq=100),
            GenerateSampleGridFigureCallback(),
        ]
    )
    exit()

    # Total number of training images seen. Used as the global step.
    step = tf.Variable(0, dtype=tf.int64)

    eval_latents = tf.random.uniform([8, gan.generator.latent_size], seed=123)

    # checkpoint setup
    checkpoint = tf.train.Checkpoint(gan=gan, step=step)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)


    # TODO: make a unique directory for each run if we're hyper-parameter tuning.
    # tensorboard setup
    train_log_dir = checkpoint_dir + '/train'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")


    print(f"Starting at step: {int(step.numpy())}")
    for image_batch in dataset:
        # number of batches
        n = int(step / batch_size)
        k_img = (step / 1000).numpy()
        
        message = "n", n, "k_img", k_img, "std:", gan.std
        
        # Train the discriminator
        d_loss = gan.discriminator_step(image_batch)
        message += "d loss", d_loss

        if n % d_steps_per_g_step == 0:
            # Train the Generator
            g_loss = gan.generator_step()
            message += "G loss:", g_loss
        
        tf.print(*message)
        
        if n % save_interval == 0:
            save_path = manager.save()
            tf.print("Saved checkpoint for step", step, "at path", save_path)

        if n % tick_interval == 0 and n != 0:
            with train_summary_writer.as_default():
                for metric in gan.metrics:
                    tf.summary.scalar(metric.name, metric.result(), step=step)
                fakes = gan.generate_samples(eval_latents)
                tf.summary.image("fakes", fakes, step=step)
        
        # increment the step.
        step.assign_add(batch_size)
