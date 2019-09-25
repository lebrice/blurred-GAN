import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
from typing import *
import dataclasses
import json

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


def locate_model_file(result_dir: str, run_name: str, suffix="hdf5") -> str:
    import glob
    import os
    paths = glob.glob(os.path.join(result_dir, f"*-{run_name}/model_*.{suffix}"))
    if not paths:
        raise FileNotFoundError

    paths = sorted(paths, key=run_id, reverse=True)
    latest_run_id = run_id(paths[0])

    paths = list(filter(lambda p: run_id(p) == latest_run_id, paths))
    paths = sorted(paths, key=epoch, reverse=True)
    return paths[0]


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


def NHWC_to_NCHW(image: tf.Tensor) -> tf.Tensor:
    return tf.transpose(image, [0, 3, 1, 2])


def NCHW_to_NHWC(image: tf.Tensor) -> tf.Tensor:
    return tf.transpose(image, [0, 2, 3, 1])


def to_dataset(t: Union[tf.Tensor, np.ndarray, tf.data.Dataset]) -> tf.data.Dataset:
    if isinstance(t, tf.data.Dataset):
        return t
    t = tf.convert_to_tensor(t)
    return tf.data.Dataset.from_tensor_slices(t)


def read_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
            return json.load(f)


from tensorflow.python.framework.ops import EagerTensor

from dataclasses import dataclass
from tensorflow.python.training.tracking.tracking import AutoTrackable

class JsonSerializable():
    def asdict(self):
        d = dataclasses.asdict(self)
        d_without_tf_objects = {}
        for k, v in d.items():
            if isinstance(v, (tf.Variable, tf.Tensor, EagerTensor)):
                d_without_tf_objects[k] = float(v.numpy())
            else:
                d_without_tf_objects[k] = v
        return d_without_tf_objects

    def save_json(self, file_path: str) -> None:
        with open(file_path, 'w') as f:
            d = self.asdict()
            json.dump(d, f, indent=1)

    @classmethod
    def from_json(cls, file_path: str):
        d = read_json(file_path)
        return cls(**d)

@dataclass
class HyperParams(AutoTrackable, JsonSerializable):
    """
    Simple wrapper for a python dataclass which enables saving and restoring from Tensorflow checkpoints. 
    Values are tracked using the `AutoTrackable` tensorflow class.
    Note: under the hood, this makes a tf.constant out of each of the values of the dataclass.
    """
    def __setattr__(self, key, value):
        v = value
        if isinstance(value, (int, float)):
            v = tf.constant(value)
        super().__setattr__(key, v) 

    def __repr__(self):
        return self.asdict().__repr__()

    def __str__(self):
        return str(self.asdict())