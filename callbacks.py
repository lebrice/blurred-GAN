import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
from blurred_gan import BlurredGAN


class GenerateSampleGridFigureCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: str, frequency=100):
        super().__init__()
        self.log_dir = log_dir
        self.frequency = frequency
        
        # provide type-hinting for the model class.
        # self.model: GAN = self.model
        # TODO: need a constant, random vector.
        self.latents = tf.random.uniform([64, 100]).numpy()
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def on_batch_end(self, batch, logs):
        if batch % self.frequency == 0:
            self.function(step=batch)

    def function(self, step):
        samples = self.model.generate_samples(self.latents, training=False)
        samples = normalize_images(samples)
        figure = samples_grid(samples)  # TODO: write figure to a file?
        image = plot_to_image(figure)
        with self.summary_writer.as_default():
            tf.summary.image("samples_grid", image, step=step)
        

@tf.function
def normalize_images(images):
    return (images + 1) / 2


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def samples_grid(samples):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure()
    for i in range(64):
        # Start next subplot.
        plt.subplot(8, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(samples[i])
    plt.tight_layout(pad=0)
    return figure

