import numpy as np
import gpflow
from matplotlib import pyplot as plt
import matplotlib
import os
import logging

matplotlib.rcParams['figure.figsize'] = (12, 6)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logging.getLogger('gpflow.logdensities').setLevel(logging.ERROR)


def evalMLE(X, Y):  # type: (Any, {shape}) -> Tuple[None, GPR]
    with gpflow.defer_build():
        k = gpflow.kernels.Matern52(1, lengthscales=0.3)
        meanf = gpflow.mean_functions.Zero()
        m = gpflow.models.GPR(X, Y, k, meanf)
        m.likelihood.variance = 0.01

    m.compile()
    print(m.as_pandas_table())

    gpflow.train.ScipyOptimizer().minimize(m)
    return None, m


def evalMCMC(X, Y):    # type: (Any, {shape}) -> Tuple[DataFrame, GPR]
    with gpflow.defer_build():
        k = gpflow.kernels.Matern52(1, lengthscales=0.3)
        meanf = gpflow.mean_functions.Zero()
        m = gpflow.models.GPR(X, Y, k, meanf)
        m.clear()

        m.kern.lengthscales.prior = gpflow.priors.Beta(1., 3.)
        m.kern.variance.prior = gpflow.priors.Beta(1., 3.)
        m.likelihood.variance.prior = gpflow.priors.Beta(1., 3.)
        # m.mean_function.A.prior = gpflow.priors.Gaussian(0., 10.)
        # m.mean_function.b.prior = gpflow.priors.Gaussian(0., 10.)

    m.compile()
    print(m.as_pandas_table())

    sampler = gpflow.train.HMC()
    traces = sampler.sample(m, num_samples=12000, burn=1000, epsilon=0.05, lmin=1, lmax=3, logprobs=False)
    return traces, m


def plot(X, Y, x, m, t, f=None, output=None):

    # GPflow models have several prediction methods:
    # * m.predict_f returns the mean and variance of the latent function (f) at the points Xnew.
    # * m.predict_f_full_cov additionally returns the full covariance matrix of the prediction.
    # * m.predict_y returns the mean and variance of a new data point (i.e. includes the noise varaince).
    #   In the case of non-Gaussian likelihoods, the variance is computed by (numerically) integrating
    #   the non-Gaussian likelihood.
    # * m.predict_f_samples returns samples of the latent function
    # * m.predict_density returns the log-density of the points Ynew at Xnew.

    mean, var = m.predict_y(x)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(x, mean, 'C0', lw=2)
    if f is not None:
        plt.plot(x, f(x), 'k', lw=2)
    plt.fill_between(x[:, 0],
                     mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                     mean[:, 0] + 2 * np.sqrt(var[:, 0]),
                     color='C0', alpha=0.2)
    plt.xlim(-0.1, 1.1)
    plt.title(t)
    if output is not None:
        plt.savefig(output)
    plt.show()
