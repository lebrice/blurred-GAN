import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import InputLayer, AveragePooling2D
from networks_keras import WeightScaledConv2D, WeightScaledDense, num_filters, MinibatchStdDevLayer, FromRGB
from utils import NCHW_to_NHWC, NHWC_to_NCHW, unnormalize_images
from typing import List, Dict, Optional, Union

import tensorflow as tf
from tensorflow import keras
from networks_keras import num_filters, Upscale2DConv2D, Upscale2D, ToRGB, PixelNorm, WeightScaledConv2D, WeightScaledDense
from utils import num_filters, resolution_of_stage, stage_of_resolution, assert_valid_resolution, log2
# tf.enable_eager_execution()

class DiscriminatorBlock(tf.keras.models.Sequential):
    def __init__(self, resolution: int, kernel_size=3, *args, **kwargs):
        self.resolution = resolution
        if "name" not in kwargs:
            kwargs["name"] = f"disc_block_{resolution}x{resolution}"
        super().__init__(
            layers=discriminator_block(resolution),
            *args,
            **kwargs
        )

def discriminator_block(resolution: int) -> List[tf.keras.layers.Layer]:
    stage = stage_of_resolution(resolution)
    return [
        tf.keras.layers.InputLayer([num_filters(stage), resolution, resolution]),
        WeightScaledConv2D(filters=num_filters(stage), kernel_size=3, name=f"disc_{stage}_conv0"),
        WeightScaledConv2D(filters=num_filters(stage), kernel_size=3, name=f"disc_{stage}_conv1"),
        AveragePooling2D(name=f"disc_{stage}_avg_pooling"),
    ]


class Discriminator(tf.keras.models.Sequential):
    def __init__(self, resolution = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert_valid_resolution(resolution)

        self.input_resolution: int = resolution
        self.stage = stage_of_resolution(self.input_resolution)

        self.layers: List[tf.keras.layers.Layer]
        self._mixing_factor: float = 0.0

        self.rgb_outputs: List[tf.keras.layers.Layer] = []
        self.blocks: List[tf.keras.layers.Layer] = []

        print(f"Building discriminator. Input resolution: {resolution}x{resolution}")
        res = resolution
        while resolution >= 4:
            print(f"Creating the block for {resolution}x{resolution}")
            filters = filters_for_resolution(resolution)
            from_rgb_layer_name = f"{resolution}x{resolution}_from_rgb"
            from_rgb_layer = FromRGB(filters, name=from_rgb_layer_name)
            
            self.rgb_outputs.append(from_rgb_layer)
            
            block = DiscriminatorBlock(resolution)
            self.blocks.append(block)

            resolution //= 2
        
        last_block = tf.keras.Sequential(layers=[
            MinibatchStdDevLayer(),
            WeightScaledConv2D(filters=num_filters(1), kernel_size=3, name="conv0"),
            WeightScaledConv2D(filters=num_filters(1), kernel_size=4, name="conv1"),
            tf.keras.layers.Flatten(),
            WeightScaledDense(units=1, gain=1, activation=tf.keras.activations.linear, name="dense0"),
        ], name="output_block")
        self.blocks.append(last_block)

    @property
    def mixing_factor(self) -> tf.Tensor:
        return tf.clip_by_value(self._mixing_factor, 0.0, 1.0)
    
    @mixing_factor.setter
    def mixing_factor(self, value) -> None:
        self._mixing_factor = value

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        print("Creating the discriminator graph. Stage is:", self.stage)
        print([l.name for l in self.blocks])

        x = self.rgb_outputs[0](inputs)
        for i, block in enumerate(self.blocks):
            if i == 0 and self.stage > 0:
                # we need the output of the first block to be slightly different:
                # it is a mix of a downscaled version of the input, and of the 
                downscaled_input = Downscale2D()(inputs)
                downscaled_features = self.rgb_outputs[1](downscaled_input)
                x = (
                    self.mixing_factor * block(x) +
                    (1-self.mixing_factor) * downscaled_features
                )
            else:
                x = block(x)
        return x
        


    
    def clear(self):
        """Remove all layers. TODO: check if this is working."""
        while len(self.layers) > 0:
            self.pop()

    def grow_new(self) -> "Discriminator":
        """
        Creates a bigger discriminator, copies the current weights over, and returns it.
        """
        print("Growing a new Discriminator, and starting it from this one's weights.")
        new = Discriminator(self.input_resolution * 2)
        current_layers = {
            l.name: l for l in self.layers 
        }
        for layer in new.layers:
            print("layer:", layer.name)
            if layer.name in current_layers:
                print(f"layer '{layer.name}' is common.")
                layer.set_weights(current_layers[layer.name].get_weights())
        return new



class GeneratorBlock(tf.keras.models.Sequential):
    def __init__(self, resolution: int, *args, **kwargs):
        self.resolution = resolution
        if "name" not in kwargs:
            kwargs["name"] = f"generator_block_{self.resolution}x{self.resolution}"
        super().__init__(
            layers=generator_block(resolution),
            *args,
            **kwargs
        )

def generator_block(resolution: int) -> List[keras.layers.Layer]:
    assert_valid_resolution(resolution)
    stage = stage_of_resolution(resolution)
    filters = num_filters(stage)
    if stage == 0:
        # First block:
        return [
            PixelNorm(name="pixelnorm_1"),
            WeightScaledDense(units=filters * 4 * 4, name="dense"),
            tf.keras.layers.Reshape([filters, 4, 4], name="reshape"),
            PixelNorm(name="pixelnorm_2"),
            WeightScaledConv2D(filters=filters, kernel_size=3, name="conv2d"),
            PixelNorm(name="output"),
        ]
    
    # later blocks
    return [
        # tf.keras.layers.InputLayer([num_filters(stage-1), res, res]),
        Upscale2DConv2D(filters=filters, kernel_size=3, name="upscale"),
        PixelNorm(name="pixelnorm"),
        WeightScaledConv2D(filters=filters, kernel_size=3, name="conv2d"),
        PixelNorm(name="output"),
    ]


class Generator(tf.keras.models.Sequential):
    def __init__(self, resolution: int, *args, **kwargs):
        assert_valid_resolution(resolution)
        super().__init__(*args, **kwargs)
        
        self.resolution = resolution

        res = 4
        while res < self.resolution:
            self.add(GeneratorBlock(res))
            res *= 2

        self.add(ToRGB())
        self.add(keras.layers.Activation(keras.activations.sigmoid))
        # flip the images to channels_last format.
        self.add(tf.keras.layers.Lambda(NCHW_to_NHWC))




