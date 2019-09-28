"""
This package is intented to contain some GAN metrics for Tensorflow 2.0.
(Mostly copied from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/)
"""


import tensorflow as tf
import numpy as np
from scipy.linalg import sqrtm
from typing import *

import sliced_wasserstein as sw
import utils
import tensorflow_hub as hub


def calculate_fid(x: np.ndarray, y: np.ndarray) -> float:
    import scipy
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)
    diff_means_squared = np.dot((mean_x - mean_y), (mean_x - mean_y).T)
    sigma_term = sigma_x + sigma_y - 2 * \
        scipy.linalg.sqrtm((sigma_x @ sigma_y))
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


def evaluate_fid(reals: np.ndarray, fakes: np.ndarray, feature_extractor: tf.keras.Model, batch_size=32):
    # assert reals.shape == fakes.shape, "shapes should match"
    # assert feature_extractor.input_shape[1:] == reals.shape[1:], "feature extractor's input doesn't match the provided data's shapes."

    # reals = utils.to_dataset(reals).batch(batch_size)
    real_features = feature_extractor(reals)
    del reals  # save some memory maybe?

    # fakes = utils.to_dataset(fakes).batch(batch_size)
    fake_features = feature_extractor(fakes)
    del fakes

    return calculate_fid_safe(real_features, fake_features)


class SWDMetric():
    """
    NOTE: Keras metrics execute in graph mode. at the moment, the code to calculate SWD is in numpy, and as such, we can't actually inherit from tf.keras.metrics.Metric.
    In the future, if this changes, then we should inherit from tf.keras.metrics.Metric. 
    """
    def __init__(self, name="SWDx1e3_avg", dtype=None):
        self.nhood_size = 7
        self.nhoods_per_image = 128
        self.dir_repeats = 4
        self.dirs_per_repeat = 128
        self.resolutions = []

    def get_metric_names(self):
        return ['SWDx1e3_%d' % res for res in self.resolutions] + ['SWDx1e3_avg']

    def get_metric_formatting(self):
        return ['%-13.4f'] * len(self.get_metric_names())

    def reset_states(self):
        for d_list in self.real_descriptors:
            d_list.clear()
        for d_list in self.fake_descriptors:
            d_list.clear()

    def update_state(self, real_minibatch, fake_minibatch, *args, **kwargs):
        if len(self.resolutions) == 0:
            res = real_minibatch.shape[1]
            while res >= 16:
                self.resolutions.append(res)
                res //= 2
            self.real_descriptors = [[] for res in self.resolutions]
            self.fake_descriptors = [[] for res in self.resolutions]

        for lod, level in enumerate(sw.generate_laplacian_pyramid(real_minibatch, len(self.resolutions))):
            desc = sw.get_descriptors_for_minibatch(
                level, self.nhood_size, self.nhoods_per_image)
            self.real_descriptors[lod].append(desc)

        for lod, level in enumerate(sw.generate_laplacian_pyramid(real_minibatch, len(self.resolutions))):
            desc = sw.get_descriptors_for_minibatch(
                level, self.nhood_size, self.nhoods_per_image)
            self.fake_descriptors[lod].append(desc)
        
    def results(self) -> Dict[str, float]:
        """
        Returns a dictionary of metrics, where each (key: value) pair corresponds to the name and value of the sliced wasserstein distance at a given level of the gaussian pyramid.
        """
        desc_reals = [sw.finalize_descriptors(d) for d in self.real_descriptors]
        desc_fakes = [sw.finalize_descriptors(d) for d in self.fake_descriptors]
        dist = [
            sw.sliced_wasserstein(dreal, dfake, self.dir_repeats, self.dirs_per_repeat)
                for dreal, dfake in zip(desc_reals, desc_fakes)
        ]
        dist = [d * 1e3 for d in dist]  # multiply by 10^3
        dist.append(np.mean(dist))
        results = dict(zip(self.get_metric_names(), dist))
        return results

    def result(self):
        """
        The tf.keras.metrics.Metric API requires a `result()` method. In our case, we just return the average as our 'result'.
        """
        results = self.results()
        average_swd_key = self.get_metric_names()[-1]
        return results[average_swd_key]


class FIDMetric():
    """
    NOTE: Keras metrics execute in graph mode. At the moment, the code to calculate FID is in numpy, and as such, we can't actually inherit from tf.keras.metrics.Metric.
    In the future, if this changes, then we should inherit from tf.keras.metrics.Metric. 
    """
    def __init__(self, name="FID"):
        self.name = name
        self.reals: List[np.ndarray] = []
        self.fakes: List[np.ndarray] = []
        self.model_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
        self.feature_extractor = hub.KerasLayer(self.model_url, output_shape=[2048], trainable=False)
    
    def update_state(self, real_minibatch, fake_minibatch, *args, **kwargs):
        self.reals.append(real_minibatch)
        self.fakes.append(fake_minibatch)

    def reset_states(self):
        self.reals.clear()
        self.fakes.clear()
    
    def result(self):
        reals = tf.concat(self.reals, axis=0)
        fakes = tf.concat(self.fakes, axis=0)
        fid = evaluate_fid(reals, fakes, self.feature_extractor)
        return fid

# a1 = tf.random.normal((32, 28, 28, 3))
# a2 = tf.random.normal((32, 28, 28, 3))
# # a2 = a1 + 0.01
# swd = calculate_swd(a1, a2)
# print(swd)
