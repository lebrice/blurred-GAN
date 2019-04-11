"""
Author: [Fabrice Normandin](www.github.com/lebrice)

Collections of `tf.keras` implementations of the layers used in [this paper](http://arxiv.org/abs/1710.10196)

The layers defined here correspond to those defined in [this file](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py),
but with all the optimizations mentioned in the paper already implemented by default.

[1](http://arxiv.org/abs/1710.10196)
"""
from typing import NamedTuple, Type, List, Tuple, Iterable, Dict
from functools import wraps

import numpy as np
import tensorflow as tf
import os
from utils import log2, HeInitializer, num_filters, only_allow_once

def RuntimeWeightScaling(layer_base_class: Type[tf.keras.layers.Layer]):
    """
    Wraps a given Keras layer base-class, adding Runtime Weight Scaling (as described in [1]) using the constant from the He initializer.
    """
    class WeightScaledVariant(layer_base_class):
        @wraps(layer_base_class)
        def __init__(self, use_wscale = True, gain = 2.0, fan_in = None, *args, **kwargs):
            self.use_wscale = use_wscale # enable this behaviour or not.
            self.gain = gain
            self.fan_in = fan_in.value if isinstance(fan_in, tf.Dimension) else fan_in
            self._kernel = None
            if self.use_wscale:
                kwargs["kernel_initializer"] = tf.keras.initializers.RandomNormal(0, 1)
            super().__init__(
                *args,
                **kwargs
            )
        
        @property
        def kernel(self) -> tf.Tensor:
            if self.use_wscale:
                return self._kernel * self.he_constant
            else:
                return self._kernel

        @kernel.setter
        def kernel(self, value) -> None:
            if self._kernel is not None:
                raise RuntimeError("Cannot set kernel attribute more than once.")
            self._kernel = value
            self.he_constant = HeInitializer.get_constant(self._kernel.shape, self.gain, self.fan_in)
   
    return WeightScaledVariant


class WeightScaledDense(RuntimeWeightScaling(tf.keras.layers.Dense)):
    def __init__(self,
                 units: int,
                 activation=tf.nn.leaky_relu,
                 use_bias=True,
                 *args, **kwargs):
        super().__init__(units=units, activation=activation,
                         use_bias=use_bias, *args, **kwargs)


class WeightScaledConv2D(RuntimeWeightScaling(tf.keras.layers.Conv2D)):
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 activation=tf.nn.leaky_relu,
                 strides=(1, 1),
                 data_format="channels_first",
                 padding="SAME",
                 use_bias=True,
                 *args, **kwargs):
        super().__init__(
            filters=filters, kernel_size=kernel_size, activation=activation,
            strides=strides, data_format=data_format, padding=padding, *args, **kwargs)


class Conv2DDownscale2D(WeightScaledConv2D):
    def __init__(self, filters: int, kernel_size: int, strides=2, *args, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size,
                         strides=strides, *args, **kwargs)


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, *args, **kwargs):
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape


class Upscale2D(tf.keras.layers.Layer):
    def __init__(self, factor: int = 2, *args, **kwargs):
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        super().__init__(*args, **kwargs)

    def call(self, x: tf.Tensor):
        if self.factor == 1:
            return x
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, self.factor, 1, self.factor])
        x = tf.reshape(x, [-1, s[1], s[2] * self.factor, s[3] * self.factor])
        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([*input_shape[:-2], *(d.value * 2 for d in input_shape[-2:])])


class Upscale2DConv2D(RuntimeWeightScaling(tf.keras.layers.Conv2DTranspose)):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides=2,
        activation=tf.nn.leaky_relu,
        padding="SAME",
        data_format="channels_first",
        *args,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            padding=padding,
            data_format=data_format,
            *args,
            **kwargs,
        )


class ToRGB(WeightScaledConv2D):
    def __init__(self, num_channels: int = 3, *args, **kwargs):
        super().__init__(
            gain=1,
            filters=num_channels,
            kernel_size=1,
            activation=tf.keras.activations.linear,
            *args,
            **kwargs,
        )


class FromRGB(WeightScaledConv2D):
    def __init__(self, filters: int, *args, **kwargs):
        super().__init__(filters=filters, kernel_size=1, *args, **kwargs)


class MinibatchStdDevLayer(tf.keras.layers.Layer):
    def __init__(self, group_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Minibatch must be divisible by (or smaller than) group_size.
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        # [NCHW]  Input shape.
        s = x.shape
        # [GMCHW] Split minibatch into M groups of size G.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])
        # [GMCHW] Cast to FP32.
        y = tf.cast(y, tf.float32)
        # [GMCHW] Subtract mean over group.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        # [MCHW]  Calc variance over group.
        y = tf.reduce_mean(tf.square(y), axis=0)
        # [MCHW]  Calc stddev over group.
        y = tf.sqrt(y + 1e-8)
        # [M111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
        # [M111]  Cast back to original data type.
        y = tf.cast(y, x.dtype)
        # [N1HW]  Replicate over group and pixels.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])
        # [NCHW]  Append as new fmap.
        return tf.concat([x, y], axis=1)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        out = input_shape.as_list()
        out[-3] += 1 # add one to the filter dimension.
        return tf.TensorShape(out)

