import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
from blurred_gan import WGANGP


def create_result_subdir(result_dir: str, run_name: str) -> str:
    import glob
    from itertools import count
    import os
    paths = glob.glob(os.path.join(result_dir, f"*-{run_name}"))
    run_ids = map(lambda p: int(os.path.basename(p).split("-")[0]), paths)
    run_id = max(run_ids, default=0) + 1
    path = os.path.join(result_dir, f"{run_id:02d}-{run_name}")
    print(f"Creating result subdir at '{path}'")
    os.makedirs(path)
    return path


def run_id(path_string):
    return int(path_string.split("/")[-2].split("-")[0])


def epoch(path_string):
    return int(path_string.split("/")[-1].split("_")[1].split(".")[0])


def locate_model_file(result_dir: str, run_name: str):
    import glob
    import os
    paths = glob.glob(os.path.join(result_dir, f"*-{run_name}/model_*.h5"))
    if not paths:
        return None

   
    paths = sorted(paths, key=run_id, reverse=True)
    latest_run_id = run_id(paths[0])

    
    paths = filter(lambda p: run_id(p) == latest_run_id, paths)
    paths = sorted(paths, key=epoch, reverse=True)
    return paths[0]


class GenerateSampleGridFigureCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: str, show_blurred_samples=True, period=100):
        super().__init__()
        self.log_dir = log_dir
        self.period = period
        self.show_blurred_samples = show_blurred_samples
        # provide type-hinting for the model class.
        # self.model: GAN = self.model
        # TODO: need a constant, random vector.
        self.latents = tf.random.uniform([64, 100]).numpy()
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def on_batch_end(self, batch, logs):
        if batch % self.period == 0:
            self.function(step=batch)

    def function(self, step):
        samples = self.model.generate_samples(self.latents, training=False)
        if self.show_blurred_samples:
            samples = self.model.blur(samples)

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
    """Return a grid of the samples images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure()
    for i in range(64):
        # Start next subplot.
        plt.subplot(8, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        x = samples[i]
        if x.shape[-1] == 1:
            x = np.reshape(x, [*x.shape[:-1]])
        plt.imshow(x)
    plt.tight_layout(pad=0)
    return figure

