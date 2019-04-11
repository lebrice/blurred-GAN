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
from gaussian_blur import gaussian_blur, GaussianBlur2D, appropriate_kernel_size, effective_resolution_reduction

print(tf.__version__)
from progan import Generator, Discriminator

df = "NHWC"


stats = lambda t: (tf.reduce_min(t), tf.reduce_mean(t), tf.reduce_max(t))


    

# blurring_layer = GaussianBlur2D(blur_std, data_format=df)

def main():
    with tf.Graph().as_default():
        _blur_std = tf.Variable(30.0, trainable=False)
        
        res_factor = tf.Variable(0.0, trainable=False)

        latents = tf.random_normal([1, 512])
        
        # Generator used for testing.
        generator = hub.Module("https://tfhub.dev/google/progan-128/1")
        # generator = Generator(128)
                
        _images = generator(latents)

        # _kernel_size = appropriate_kernel_size(_blur_std, _images.shape[1].value)
        _blurred_images = effective_resolution_reduction(
            _images,
            res_factor,
        )
        # _blurred_images = gaussian_blur(_images, _blur_std, kernel_size=_kernel_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # get a latent vector
            latent_vector = sess.run(latents)
                        
            original = sess.run(_images, feed_dict={
                latents: latent_vector,
                # _kernel_size: 3,
                # _blur_std: 0.1,
                res_factor: 0.0,
                })


           

            # sess.run(_blur_std.assign(32))
            # factor = 1.0


            fig, axeslist = plt.subplots(ncols=5, nrows=2)

            for i, factor in enumerate(np.arange(1, 0, -0.1)):
                # blurred_image, std, k = sess.run(
                blurred_image, res = sess.run(
                    # [_blurred_images, _blur_std, _kernel_size],
                    [_blurred_images, res_factor],
                    feed_dict={
                        latents: latent_vector,
                        # _kernel_size: 5,
                        # _blur_std: 1.0,
                        res_factor: factor
                    })

                axeslist.ravel()[i].imshow(blurred_image[0])
                # axeslist.ravel()[i].set_title(f"\u03C3={std}, k={k}")
                axeslist.ravel()[i].set_title("original" if i == 0 else f"factor: {res:.2%}")
                axeslist.ravel()[i].set_axis_off()

                # sess.run(_blur_std.assign(_blur_std * 0.5))
                # factor /= 2.0
                # sess.run(res_factor.assign(res_factor * 0.5))


            plt.tight_layout() # optional
            plt.show()


if __name__ == "__main__":
    main()