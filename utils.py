import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
from typing import *
import dataclasses
import json
import argparse
from collections import defaultdict

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
        return cls(**d) #type: ignore


class ParseableFromCommandLine:
    """
    When applied to a dataclass, this enables creating an instance of that class and populating the attributes from the command-line.
    
    Example:
    ```
    @dataclass
    class Options(ParseableFromCommandLine):
        a: int
        b: int = 10

    parser = argparse.ArgumentParser()
    Options.add_cmd_args(parser)

    args = parser.parse_args("--a 5")
    options = Options.from_args(args)
    print(options) # gives "Options(a=5, b=10)"

    args = parser.parse_args("--a 1 2 --b 9")
    options_list = Options.from_args_multiple(args, 2)
    print(options_list) # gives "[Options(a=1, b=9), Options(a=2, b=9)]"

    ```
    """
    
    class InconsistentArgumentError(RuntimeError):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    @classmethod
    def add_cmd_args(cls, parser: argparse.ArgumentParser):
        """
        Adds corresponding command-line arguments for this class to the given parser.
        
        Arguments:
            parser {argparse.ArgumentParser} -- The base argument parser to use
        """
        group = parser.add_argument_group(cls.__qualname__)
        for f in dataclasses.fields(cls):
            arg_options: Dict[str, Any] = {
                "type": f.type,
                "nargs": "*",
            }
            if f.default is dataclasses.MISSING:
                arg_options["required"] = True
            else:
                arg_options["default"] = f.default
            group.add_argument(f"--{f.name}", **arg_options)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> object:
        """Creates an instance of this class using results of `parser.parse_args()`
        
        Arguments:
            args {argparse.Namespace} -- The result of a call to `parser.parse_args()`
        
        Returns:
            object -- an instance of this class
        """
        args_dict = vars(args) 
        constructor_args: Dict[str, Any] = {
            f.name: args_dict[f.name] for f in dataclasses.fields(cls) 
        }
        return cls(**constructor_args) #type: ignore

    @classmethod
    def from_args_multiple(cls, args: argparse.Namespace, num_instances_to_parse: int) -> List[object]:
        """Parses multiple instances of this class from the command line, and returns them.
        Each argument may have either 0 values (when applicable), 1, or {num_instances_to_parse}. 
        NOTE: If only one value is provided, every instance will be populated with the same value.

        Arguments:
            args {argparse.Namespace} -- The
            num_instances_to_parse {int} -- Number of instances that are to be created from the given parsedarguments
        
        Raises:
            cls.InconsistentArgumentError: [description]
        
        Returns:
            List -- A list of populated instances of this class.
        """
        args_dict: Dict[str, Any] = vars(args)
        # keep the arguments and values relevant to this class.
        constructor_arguments: Dict[str, Union[Any, List]] = {
            f.name: args_dict[f.name]
            for f in dataclasses.fields(cls)
        }

        for field_name, values in constructor_arguments.items():
            if isinstance(values, list):
                if len(values) not in {1, num_instances_to_parse}:
                    raise cls.InconsistentArgumentError(
                        f"The field {field_name} contains {len(values)} values, but either 1 or {num_instances_to_parse} values were expected.")
                if len(values) == 1:
                    constructor_arguments[field_name] = values[0]

        # convert from a dict of lists to a list of dicts.
        arguments_per_instance: List[Dict[str, Any]] = [
            {
                field_name: field_values[i] if isinstance(field_values, list) else field_values
                for field_name, field_values in constructor_arguments.items()
            } for i in range(num_instances_to_parse) 
        ]
        return [
            cls(**arguments_dict) #type: ignore
            for arguments_dict in arguments_per_instance
        ]

    @classmethod
    def read_from_command_line(cls):
        parser = argparse.ArgumentParser()
        cls.add_cmd_args(parser)
        args = parser.parse_args()
        return cls.from_args(args)
    


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


@dataclass
class Options(ParseableFromCommandLine):
    a: int
    b: int = 10

parser = argparse.ArgumentParser()
Options.add_cmd_args(parser)
args = parser.parse_args([])
options = Options.from_args_multiple(args, 2)
print(options)


