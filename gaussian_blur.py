"""
2D Gaussian Blur Keras layer.
"""
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import tensorflow.math as math
from typing import Tuple

def effective_resolution_reduction(images: tf.Tensor, factor: float) -> tf.Tensor:
    """
    reduces the "effective resolution" of the image by `factor`, using Gaussian Blur.

    image: tf.Tensor
    factor: float in [0, 1)

    TODO: validate the math here. I'm making this up based on my intuition at the moment.
    """

    h, w, c = get_image_dims(images)
    full_resolution = math.maximum(h, w)
    effective_resolution = factor * tf.cast(full_resolution, tf.float32)
    std = - 10 * math.log(factor)

    std = tf.clip_by_value(std, 1e-2, 0.5 * tf.cast(full_resolution, float))
    kernel_size = appropriate_kernel_size(std, full_resolution)
    with tf.control_dependencies([tf.print(std, kernel_size)]):
        return gaussian_blur(images, std, kernel_size)



def appropriate_kernel_size(std: float, image_res: int) -> int:
    """
    Returns the appropriate gaussian kernel size to be used for a given standard deviation.
    """
    # no need to reach more than 3 std's to either side of the middle:
    size = math.ceil(std * 6)
    size = (size * 2) // 2 + 1.0  # nearest odd number.
    size = math.minimum(math.maximum(size, 3), tf.cast(image_res, tf.float32))
    print("std: ", std, "size:", size)
    return size


def get_data_format(image) -> str:
    if image.shape[-1].value in (1, 3):
        return "NHWC"
    else:
        return "NCHW"


def get_image_dims(image) -> Tuple[int, int]:
    data_format = get_data_format(image)
    image_height = image.shape[1 if data_format == "NHWC" else -2]
    image_width = image.shape[2 if data_format == "NHWC" else -1]
    image_channels = image.shape[-1 if data_format == "NHWC" else 1]
    return image_height, image_width, image_channels


# @tf.function
def gaussian_blur(
    image: tf.Tensor,
    std: float,
    kernel_size: int = None
):
    """
    Performs gaussian blurring.

    NOTE: Since the gaussian filter is separable, we use a 1d kernel and
    convolve twice (more efficient).
    """
    data_format = get_data_format(image)
    assert data_format in {"NHWC", "NCHW"}, "invalid data format"
    
    h, w, c = get_image_dims(image)

    if kernel_size is None:
        size = appropriate_kernel_size(std, image_width.value)
    else:
        size = kernel_size

    distribution = tfp.distributions.Normal(0, std)
    vals = distribution.prob(tf.range(-(size//2), (size//2)+1, dtype=float))
    kernel = vals / tf.reduce_sum(vals)
    
    summary = tf.summary.image(
        "gaussian_kernel",
        tf.einsum("i,j->ij", kernel, kernel)[tf.newaxis, :, :, tf.newaxis]
    )

    # expand the kernel to match the requirements of depthsiwe_conv2d
    kernel = kernel[:, tf.newaxis, tf.newaxis, tf.newaxis]
    kernel_h = tf.tile(kernel, [1, 1, c, 1])
    kernel_v = tf.transpose(kernel_h, [1, 0, 2, 3])
        
    result_1 = tf.nn.depthwise_conv2d(
        image,
        kernel_h,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format=data_format,
    )
    # flip the kernel, so it is now vertical
    result_2 = tf.nn.depthwise_conv2d(
        result_1,
        kernel_v,
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format=data_format,
    )

    return result_2


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    kernel_2d = tf.einsum('i,j->ij', kernel_1, kernel_2)
    # Make data.
    X = np.arange(0, size, 1)
    Y = np.arange(0, size, 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, kernel, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()