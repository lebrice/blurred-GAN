
import tensorflow as tf
import tensorflow_datasets as tfds


def celeba_dataset(batch_size=32, shuffle_buffer_size=100) -> tf.data.Dataset:
    """Modern Tensorflow input pipeline for the CelebA dataset"""

    @tf.function
    def take_image(example):
        return example["image"]

    @tf.function
    def normalize(image):
        return (tf.cast(image, tf.float32) - 127.5) / 127.5

    @tf.function
    def resize_image(image):
        image = tf.image.resize(image, [128, 128])
        return image

    @tf.function
    def preprocess_images(image):
        image = normalize(image)
        image = resize_image(image)
        return image

    celeba_dataset = tfds.load(name="celeb_a", split=tfds.Split.ALL)
    celeba = (celeba_dataset
              .map(take_image)
              .batch(16)
              .map(preprocess_images)
            #   .cache("./cache")
              .apply(tf.data.experimental.unbatch())
              .shuffle(shuffle_buffer_size)
              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
              .batch(batch_size)
              )
    return celeba
