"""
Keras callbacks used to modify the gaussian blur's std during training.
"""
import tensorflow as tf
import metrics

class BlurScheduleController(tf.keras.callbacks.Callback):
    def __init__(self, total_n_training_batches: int, max_value: float = 23.5, min_value=0.01):
        # if schedule_type == "exponential_decay":
        self.schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            float(max_value),
            decay_steps=total_n_training_batches / 10, # leave some more 'fine-tuning' time near the end.
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
        std_was_just_modified = batch - self._last_modification_step < self.delay_between_modifications
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
        self.score_ratio = self.smoothing * self.score_ratio + (1 - self.smoothing) * ratio
        
        if batch < self.warmup_n_batches:
            return
        
        with self.model.summary_writer.as_default():
            tf.summary.scalar("blur_controller/ratio", ratio)
            tf.summary.scalar("blur_controller/smoothed_ratio", self.score_ratio)
            tf.summary.scalar("blur_controller/stable", int(self.gan_problem_is_stable()))

        if self.gan_problem_is_stable():
           # print(f"\nProblem is too easy. (ratio is currently {ratio}) reducing the blur std (currently {self.std})")
           self.decrease_blur_std(batch)


        if self.std < self.min_value:
            print("Reached the minimum STD. Training is complete.")
            self.model.stop_training = True


class FIDScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_preprocessing_fn, dataset_fn, n=1000):
        self.image_preprocessing_fn = image_preprocessing_fn
        self.make_dataset = dataset_fn
        self.n = n
    

    def on_epoch_end(self, epoch, logs):
        fid = self.fid_score()
        print(f"\nFID: {fid}")

        with self.model.summary_writer.as_default():
            tf.summary.scalar("fid_score", fid)

    def fid_score(self) -> float:
        import tensorflow_hub as hub
        model_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

        feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Lambda(self.image_preprocessing_fn),
            hub.KerasLayer(model_url, output_shape=[2048], input_shape=[299,299,3], trainable=False),
        ])
        
        reals = self.make_dataset().take(self.n)
        fakes = self.model.generator.predict(tf.random.uniform((self.n, self.model.generator.input_shape[-1])))
        
        return metrics.evaluate_fid(reals, fakes, feature_extractor)