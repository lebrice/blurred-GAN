"""
Progressively Growing GANS using Gaussian Blur
"""
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_probability as tfp
import tensorflow.keras as keras
import tensorflow.math as math

import numpy as np
import matplotlib.pyplot as plt

from gaussian_blur import GaussianBlur2D




generator_module_url = "https://tfhub.dev/google/progan-128/1"

with tf.Graph().as_default():
    # Generate 20 random samples.
    generate = hub.Module(generator_module_url)
    images = generate(tf.random_normal([20, 512]))
    plt.imshow(images[0])




