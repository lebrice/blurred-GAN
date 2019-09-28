import tensorflow as tf
import matplotlib.pyplot as plt
from typing import *
from enum import Enum
import dataclasses
from dataclasses import dataclass, InitVar, field
import os
import json
from contextlib import contextmanager

from gaussian_blur import GaussianBlur2D
from utils import JsonSerializable, ParseableFromCommandLine

from wgan import WGAN, WGANGP, TrainingConfig


def BlurredVariant(some_gan_base_class):
    class BlurredGAN(some_gan_base_class):
        """
        IDEA: Simple variation on the WGAN-GP (or any GAN architecture, for that matter) where we added the blurring layer in the discriminator.
        TODO:
        Maybe the hyperparameter classes could also be inside the corresponding GAN classes..
        """

        @dataclass
        class HyperParameters(some_gan_base_class.HyperParameters):
            initial_blur_std: float = 0.05

        def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, hyperparams: HyperParameters, config: TrainingConfig, **kwargs):
            blur = GaussianBlur2D(initial_std=hyperparams.initial_blur_std, input_shape=discriminator.input_shape[1:])
            discriminator_with_blur = tf.keras.Sequential([
                blur,
                discriminator,
            ])
            super().__init__(generator, discriminator_with_blur, hyperparams=hyperparams, config=config, **kwargs)
            self.blur = blur
            self.std_metric = tf.keras.metrics.Mean("std", dtype=tf.float32)
        
        @property
        def std(self):
            return self.blur.std

        def discriminator_step(self, reals):
            with self.record_image_summaries():
                disc_loss, images = super().discriminator_step(reals)

            self.std_metric(self.std) # add this 'std' metric
            return disc_loss, images
    return BlurredGAN

BlurredWGANGP = BlurredVariant(WGANGP)
BlurredWGAN = BlurredVariant(WGAN)
