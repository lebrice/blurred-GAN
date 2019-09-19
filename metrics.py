"""
This package is intented to contain some GAN metrics for Tensorflow 2.0.
(Mostly copied from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/)
"""


import tensorflow as tf
import numpy as np
from scipy.linalg import sqrtm
from typing import *

def calculate_fid(x: np.ndarray, y: np.ndarray) -> float:
    import scipy
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)
    diff_means_squared = np.dot((mean_x - mean_y), (mean_x - mean_y).T)
    sigma_term = sigma_x + sigma_y - 2 * scipy.linalg.sqrtm((sigma_x @ sigma_y))
    if np.iscomplexobj(sigma_term):
        sigma_term = sigma_term.real
    return diff_means_squared + np.trace(sigma_term)

def covariance(x):
    """
    Copied directly from https://stackoverflow.com/questions/47709854/how-to-get-covariance-matrix-in-tensorflow?rq=1 
    """
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx

def calculate_fid_safe(act1: np.ndarray, act2: np.ndarray, epsilon=1e-6) -> np.ndarray:
    """
    Copied directly from https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    """
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % epsilon
        # warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * epsilon
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def to_dataset(t: Union[tf.Tensor, np.ndarray, tf.data.Dataset]) -> tf.data.Dataset:
        if isinstance(t, tf.data.Dataset):
            return t
        t = tf.convert_to_tensor(t)
        return tf.data.Dataset.from_tensor_slices(t)


def evaluate_fid(reals: np.ndarray, fakes: np.ndarray, feature_extractor: tf.keras.Model, batch_size=32):
    # assert reals.shape == fakes.shape, "shapes should match"
    # assert feature_extractor.input_shape[1:] == reals.shape[1:], "feature extractor's input doesn't match the provided data's shapes."
 
    reals = to_dataset(reals).batch(batch_size)
    real_features = feature_extractor.predict(reals)
    del reals # save some memory maybe?
    
    fakes = to_dataset(fakes).batch(batch_size)
    fake_features = feature_extractor.predict(fakes)
    del fakes

    return calculate_fid_safe(real_features, fake_features)

import sliced_wasserstein_impl
from sliced_wasserstein_impl import sliced_wasserstein_distance

def mean_sliced_wasserstein_distance(real_images, fake_images):
    distances = sliced_wasserstein_distance(real_images, fake_images)
    real_distances, fake_distances = [], []
    for i, (distance_real, distance_fake) in enumerate(distances):
        print(f"level: {i}, distance_real: {distance_real}, distance_fake: {distance_fake}")
        real_distances.append(distance_real)
        fake_distances.append(distance_fake)
    return tf.reduce_mean(fake_distances)


# a1 = tf.random.normal((32, 28, 28, 3))
# a2 = tf.random.normal((32, 28, 28, 3))
# # a2 = a1 + 0.01
# swd = calculate_swd(a1, a2)
# print(swd)
