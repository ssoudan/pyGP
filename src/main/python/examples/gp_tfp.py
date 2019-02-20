import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
import os

from .data.gen import make_data
import gp.gp_tfp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# see https://github.com/tensorflow/probability/blob/master/tensorflow_probability/
#             g3doc/api_docs/python/tfp/distributions/GaussianProcessRegressionModel.md
#
tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels


def evalGPRSample():  # type: () -> None

    # Generate noisy observations from a known function at some random points.
    observation_noise_variance = .5

    def f(x):
        return np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)

    observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
    observations = f(observation_index_points) + np.random.normal(0., np.sqrt(observation_noise_variance))
    index_points = np.linspace(-1., 1., 100)[..., np.newaxis]

    kernel = psd_kernels.MaternFiveHalves()

    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance)

    samples = gprm.sample(10)
    print(samples)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        samples_ = sess.run(samples)
        print(samples_)


def run(output="output/"):
    evalGPRSample()

    X, Y, x, f = make_data()

    _, y = gp.gp_tfp.evalMLE(X, Y, x)

    gp.gp_tfp.plot(X, Y, x, y, f, title="MLE", output=os.path.join(output, "tfp_mle.png"))

    _, y = gp.gp_tfp.evalHMC(X, Y, x)
    gp.gp_tfp.plot(X, Y, x, y, f, title="HMC", output=os.path.join(output, "tfp_hmc.png"))
