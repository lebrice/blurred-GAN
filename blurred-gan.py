#!/usr/bin/python3

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from enum import Enum
print("Hello")
print(tf.__version__)

tfp.distributions


class DataFormat(Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"


def gaussian_kernel(size: int, mean: float, std: float) -> tf.Tensor:
    """
    Makes 2D gaussian Kernel for convolution.
    (Taken from https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow)
    """

    d = tfp.distributions.Normal(mean, std)

    vals = tf.random.normal(shape=[size, size, 1, 1], )
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    return 


class GaussianBlur(keras.layers.Layer):
    def __init__(self, size: int, std: float, data_format=DataFormat.NCHW):
        super().__init__()
        self.size = size
        self.std = std
        # self.conv2d = tf.keras.layers.Conv2D()
        self.data_format = data_format
        self.kernel = gaussian_kernel(size=size, mean=0, std=std)
        self.kernel = self.kernel[:, :, tf.newaxis, tf.newaxis]

    def call(self, image: tf.Tensor, training=True):
        # Convolve.
        return tf.nn.conv2d(
            image,
            self.kernel,
            strides=[1, 1, 1, 1],
            data_format=self.data_format,
            padding="SAME"
        )


bob = GaussianBlur(5, 1.0)