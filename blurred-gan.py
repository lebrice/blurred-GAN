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


tf.enable_eager_execution()
from gaussian_blur import gaussian_blur, GaussianBlur2D, appropriate_kernel_size

print(tf.__version__)
from progan import Generator, Discriminator

df = "NHWC"


stats = lambda t: (tf.reduce_min(t), tf.reduce_mean(t), tf.reduce_max(t))


# blurring_layer = GaussianBlur2D(blur_std, data_format=df)

def main():
    with tf.Graph().as_default():
        _blur_std = tf.Variable(10.0, trainable=False)
        
        latents = tf.random_normal([1, 512])
        
        # Generator used for testing.
        generator = hub.Module("https://tfhub.dev/google/progan-128/1")
        # generator = Generator(128)
        
        _images = generator(latents)

        _kernel_size = appropriate_kernel_size(_blur_std, _images.shape[1].value)
        with tf.control_dependencies([tf.print("kernel_size:", _kernel_size)]):
            _blurred_images = gaussian_blur(_images, _blur_std, kernel_size=_kernel_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # get a latent vector
            latent_vector = sess.run(latents)
                        
            original = sess.run(_images, feed_dict={latents: latent_vector})

            fig, axeslist = plt.subplots(ncols=3, nrows=2)      
            for i in range(5):
                blurred_image, std, k = sess.run([_blurred_images, _blur_std, _kernel_size], feed_dict={
                    latents: latent_vector,
                    _kernel_size: 11
                })

                axeslist.ravel()[i].imshow(blurred_image[0])
                axeslist.ravel()[i].set_title(f"\u03C3={std}, k={k}")
                axeslist.ravel()[i].set_axis_off()

                sess.run(_blur_std.assign(_blur_std * 0.5))

            axeslist.ravel()[-1].imshow(original[0])
            axeslist.ravel()[-1].set_title("Original")
            axeslist.ravel()[-1].set_axis_off()

            plt.tight_layout() # optional
            plt.show()


if __name__ == "__main__":
    main()