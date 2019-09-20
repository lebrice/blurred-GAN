"""
Keras callbacks used to modify the gaussian blur's std during training.
"""
import tensorflow as tf
import tensorflow_hub as hub
import metrics
from typing import *
import numpy as np
import utils


class ExecuteEveryNExamplesCallback(tf.keras.callbacks.Callback):
    """
    Executes a given function approximately every N examples, depending on if the period is an even multiple of the batch size or not.
    """

    def __init__(self, period: int, starting_from: int = 0):
        """
        args:
            n: executes the `self.function(batch, logs)` method approximately every N examples
            starting_from: The first invocation should occur after this number of examples (defaults to 0)
        """
        super().__init__()
        self.period = period
        self.num_invocations = 0
        self.samples_seen = 0
        self.starting_from = starting_from

    def on_batch_end(self, batch, logs: Dict):
        batch_size = logs["size"]
        self.samples_seen += batch_size
        i = (self.samples_seen - self.starting_from) // self.period
        # print("\n", i, self.samples_seen, self.starting_from, self.period)
        if self.samples_seen < self.starting_from:
            return
        if i >= self.num_invocations:
            self.num_invocations += 1
            # print(f"\nsamples_seen: {self._samples_seen}, batch: {batch}, i: {self.i}\n")
            # TODO: Check the function signature.
            self.function(batch, logs)

    def function(self, batch, logs):
        raise NotImplementedError("Implement the 'function' inside your class!")

class BlurDecayController(tf.keras.callbacks.Callback):
    """
    TODO: rework this.
    """
    def __init__(self, total_n_training_examples: int, max_value: float = 23.5, min_value=0.01):
        # if schedule_type == "exponential_decay":
        self.schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            float(max_value),
            # leave some more 'fine-tuning' time near the end.
            decay_steps=total_n_training_examples / 10,
            decay_rate=0.96,
            staircase=False,
        )
        # elif schedule_type == (...)

    def on_batch_begin(self, batch, logs):
        value = self.schedule(self.model.n_batches)
        self.model.std.assign(value)


class AdaptiveBlurController(tf.keras.callbacks.Callback):
    """
    Controller which adaptively reduces the amount of blurring used during
    training. To be used with the `BlurredGAN` keras model.

    Once the standard deviation reaches a value equal to `min_value`, the
    training stops.
    """

    def __init__(self, smoothing=0.99, warmup_n_batches=100, threshold=0.05, min_value=0.01, max_value=23.5):
        super().__init__()
        self.smoothing = smoothing
        self.warmup_n_batches = warmup_n_batches
        self.score_ratio = 0.5
        self.threshold = threshold

        self._last_modification_step = 0
        self.delay_between_modifications = 100

        self.std = float(max_value)
        self.min_value = min_value

    def on_train_begin(self, logs=None):
        self.model.std.assign(self.std)

    def gan_problem_is_stable(self) -> bool:
        min_threshold = 0.5 - self.threshold
        max_threshold = 0.5 + self.threshold
        return min_threshold <= self.score_ratio <= max_threshold

    def decrease_blur_std(self, batch: int) -> None:
        # TODO: Test this
        std_was_just_modified = batch - \
            self._last_modification_step < self.delay_between_modifications
        if not std_was_just_modified:
            self.std = self.smoothing * self.std

            # TODO: re-add this step when we feel confident to use it.
            # self.model.blur.std.assign(self.std)

            with self.model.summary_writer.as_default():
                tf.summary.scalar("blur_controller/would_modify", 1)
            self._last_modification_step = batch
        else:
            with self.model.summary_writer.as_default():
                tf.summary.scalar("blur_controller/would_modify", 0)

    def on_batch_end(self, batch, logs):
        fake_scores = logs["fake_scores"]
        real_scores = logs["real_scores"]
        ratio = fake_scores / (real_scores + fake_scores)
        self.score_ratio = self.smoothing * \
            self.score_ratio + (1 - self.smoothing) * ratio

        if batch < self.warmup_n_batches:
            return

        with self.model.summary_writer.as_default():
            tf.summary.scalar("blur_controller/ratio", ratio)
            tf.summary.scalar(
                "blur_controller/smoothed_ratio", self.score_ratio)
            tf.summary.scalar("blur_controller/stable",
                              int(self.gan_problem_is_stable()))

        if self.gan_problem_is_stable():
            # print(f"\nProblem is too easy. (ratio is currently {ratio}) reducing the blur std (currently {self.std})")
            self.decrease_blur_std(batch)

        if self.std < self.min_value:
            print("Reached the minimum STD. Training is complete.")
            self.model.stop_training = True


class FIDScoreCallback(ExecuteEveryNExamplesCallback):
    """
    Calculate the FID score after every epoch. 
    """
    def __init__(self, image_preprocessing_fn: Callable[[tf.Tensor], tf.Tensor], dataset_fn: Callable[[], tf.data.Dataset], n=1000, every_n_examples=10_000,):
        self.image_preprocessing_fn = image_preprocessing_fn
        self.make_dataset = dataset_fn
        self.n = n

        self.real_samples = self.make_dataset().shuffle(1000).repeat()

        self.model_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Lambda(self.image_preprocessing_fn),
            hub.KerasLayer(self.model_url, output_shape=[2048], input_shape=[
                           299, 299, 3], trainable=False),
        ])
        super().__init__(every_n_examples)

    def function(self, batch, logs):
        self.fid_score()

    def fid_score(self, *args) -> None:
        reals = self.real_samples.take(self.n)
        fakes = self.model.generator.predict(tf.random.uniform(
            (self.n, self.model.generator.input_shape[-1])))
        fid = metrics.evaluate_fid(reals, fakes, self.feature_extractor)
        with self.model.summary_writer.as_default():
            tf.summary.scalar("fid_score", fid)
        print(f"- FID: {fid:.4f}")


class SWDCallback(ExecuteEveryNExamplesCallback):
    """
    Accumulates examples during training and calculates the SWD between real and fake images periodically.
    """
    def __init__(self, image_preprocessing_fn, n=1000, every_n_examples=10_000):
        super().__init__(period=every_n_examples, starting_from=-n)
        self.n = n
        self.image_preprocessing_fn = image_preprocessing_fn
        self.recording = False # hack to do a first run for t=0.
        self.samples_recorded = 0

        # we need the image size, which we get once trainign starts (on_train_begin below.)
        self.swd_metric: metrics.SWDMetric

    def on_train_begin(self, logs):
        image_shape = self.model.discriminator.input_shape[1:]
        self.swd_metric = metrics.SWDMetric(image_shape)
    
    def function(self, batch, logs):
        self.recording = True
    
    def on_batch_end(self, batch: int, logs: Dict):
        super().on_batch_end(batch, logs)
        if self.recording:            
            fakes, reals, blurred_fakes, blurred_reals = self.model.images

            # take only the number of examples we need.
            # for example, if we already have 32 examples, and the metric function expects 50 examples (i.e, n is 50), we only take 18, rather than another 32.
            batch_size: int = logs["size"]
            num_examples_recorded = min(batch_size, self.n - self.samples_recorded)
            fakes = fakes[:num_examples_recorded]
            reals = reals[:num_examples_recorded]
            reals = self.image_preprocessing_fn(reals)
            fakes = self.image_preprocessing_fn(fakes)
            self.swd_metric.update_state(reals, fakes)

            
            self.samples_recorded += num_examples_recorded # add the batch size
            if self.samples_recorded >= self.n:
                assert self.samples_recorded == self.n

                results = self.swd_metric.results()
                print(" - " + " - ".join([f"{name}: {value:.4f}" for name, value in results.items()]))
                with self.model.summary_writer.as_default():
                    for name, value in results.items():
                        tf.summary.scalar(f"swd/{name}", value)
                # stop recording now.
                self.recording = False
                self.swd_metric.reset_states()
                self.samples_recorded = 0


class GenerateSampleGridCallback(ExecuteEveryNExamplesCallback):
    def __init__(self, log_dir: str, show_blurred_samples=True, every_n_examples=1000):
        self.log_dir = log_dir
        self.show_blurred_samples = show_blurred_samples
        super().__init__(every_n_examples)

        # we need a constant random vector which will not change over the course of training. 
        self.latents: np.ndarray = None

    def function(self, batch, logs):
        self.make_grid()

    def on_train_begin(self, logs: Dict):
        self.latents = tf.random.uniform([64, self.model.generator.input_shape[-1]])

    def make_grid(self, *args):
        samples = self.model.generate_samples(self.latents, training=False)
        if self.show_blurred_samples:
            samples = self.model.blur(samples)

        samples = utils.normalize_images(samples)
        figure = utils.samples_grid(samples)  # TODO: write figure to a file?
        image = utils.plot_to_image(figure)
        with self.model.summary_writer.as_default():
            tf.summary.image("samples_grid", image)
