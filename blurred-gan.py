import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import layers
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

        self.real_scores = tf.keras.metrics.Mean("real_scores", dtype=tf.float32)
        self.fake_scores = tf.keras.metrics.Mean("fake_scores", dtype=tf.float32)
        self.gp_term = tf.keras.metrics.Mean("gp_term", dtype=tf.float32)


        self.gen_loss = tf.keras.metrics.Mean("gen_loss", dtype=tf.float32)
        self.disc_loss = tf.keras.metrics.Mean("disc_loss", dtype=tf.float32)

    def latents_batch(self):
        assert self.batch_size is not None
        return tf.random.uniform([self.batch_size, self.generator.latent_size])

    def generate_samples(self, latent_vector=None, training=False):
        if latent_vector is None:
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

    # @Function
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

        discriminator_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))
        
        return disc_loss
    
    # @Function
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


def celeba_dataset(batch_size=32, epochs=None, shuffle_buffer_size=100) -> tf.data.Dataset:
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
              .apply(tf.data.experimental.unbatch())
              .shuffle(shuffle_buffer_size)
              .batch(batch_size)
              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
              .repeat(epochs)
              )
    return celeba


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import datetime
    gan = GAN()

    EPOCHS = 10
    batch_size = 16
    d_steps_per_g_step = 3
    save_interval = 50

    # interval at which tensorboard summaries should be saved
    tick_interval = 100

    dataset = celeba_dataset(batch_size=batch_size)

    results_dir = "./results"
    checkpoint_dir = results_dir + "/testrun"

    # number of batches. TODO: change to number of examples seen. 
    step = tf.Variable(1)

    eval_latents = tf.random.uniform([8, gan.generator.latent_size], seed=123)

    # checkpoint setup
    checkpoint = tf.train.Checkpoint(step=step, gan=gan)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    # tensorboard setup
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = checkpoint_dir + '/logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    print(f"Starting at step: {int(step.numpy())}")
    for image_batch in dataset:
        _step = int(step.numpy())

        message = f"step: {_step}, std: {gan.std.numpy()}"
        # Train the discriminator
        d_loss = gan.discriminator_step(image_batch)

        message += f"\t d loss: {d_loss.numpy()}"
        if _step % d_steps_per_g_step == 0:
            # Train the Generator
            g_loss = gan.generator_step()
            message += f"\t G loss: {g_loss.numpy()}"
        
        print(message)

        if _step % save_interval == 0:
            save_path = manager.save()
            print(f"Saved checkpoint for step {step}: {save_path}")

        if _step % tick_interval == 0:
            with train_summary_writer.as_default():
                for metric in gan.metrics:
                    tf.summary.scalar(metric.name, metric.result(), step=_step)
                fakes = gan.generate_samples(eval_latents)
                tf.summary.image("fakes", fakes)
        
        # increment the step.
        step.assign_add(1)
