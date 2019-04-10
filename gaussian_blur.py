"""
2D Gaussian Blur Keras layer.
"""
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import tensorflow.math as math


def appropriate_kernel_size(std: float, image_width: int) -> int:
    """
    Returns the appropriate gaussian kernel size to be used for a given standard deviation.
    """
    # no need to reach more than 3 std's to either side of the middle:
    size = (2 * math.ceil(std * 3 - 1)) + 1
    size = math.minimum(math.maximum(size, 3), image_width)
    print("std: ", std, "size:", size)
    return size


# @tf.function
def gaussian_blur(
    image: tf.Tensor,
    std: float,
    kernel_size: int = None,
    data_format=None,
    show_kernel=False,
):
    if data_format is None:
        if image.shape[-1] in (1, 3):
            data_format = "NHWC"
        else:
            data_format = "NCHW"
    assert data_format in {"NHWC", "NCHW"}
    
    image_width = image.shape[2 if data_format == "NHWC" else -1]
    image_channels = image.shape[-1 if data_format == "NHWC" else 1]
    
    if kernel_size is None:
        size = appropriate_kernel_size(std, image_width.value)
    else:
        size = kernel_size

    d = tfp.distributions.Normal(0, std)
    vals = d.prob(tf.range(-(size//2), (size//2)+1, dtype=tf.float32))
    kernel = tf.einsum('i,j->ij', vals, vals)
    
    if tf.executing_eagerly() and show_kernel:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        # Make data.
        X = np.arange(0, size, 1)
        Y = np.arange(0, size, 1)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, kernel, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()

    kernel /= tf.reduce_sum(kernel)  # normalize the kernel

    # expand dims for conv2d operator
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, image_channels, 1])

    return tf.nn.depthwise_conv2d(
        image,
        kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format=data_format,
    )


class GaussianBlur2D(keras.layers.Layer):
    def __init__(
        self,
        std: float,
        data_format="channels_first",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.std = std
        self.data_format = data_format
        
        self.trainable = False
        
#     @tf.function
    def call(self, image: tf.Tensor):
        flipped = tf.transpose(image, [0,2,3,1])
        tf.summary.image("image", flipped)
        return gaussian_blur(
            image,
            std=self.std,
            #data_format=self.data_format,
        )
