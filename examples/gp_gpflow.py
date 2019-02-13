import gpflow
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12, 6)

#
# See http://127.0.0.1:8888/notebooks/src/chapGP/regression.ipynb
#

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_data():
    N = 12
    X = np.random.rand(N, 1)
    Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + np.random.randn(N, 1) * 0.1 + 3

    return X, Y

def gpr(X, Y):
    k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    m = gpflow.models.GPR(X, Y, kern=k)
    m.likelihood.variance = 0.01

    return m

def plot(X, Y, m):

    # GPflow models have several prediction methods:
    # * m.predict_f returns the mean and variance of the latent function (f) at the points Xnew.
    # * m.predict_f_full_cov additionally returns the full covariance matrix of the prediction.
    # * m.predict_y returns the mean and variance of a new data point (i.e. includes the noise varaince). In the case of non-Gaussian likelihoods, the variance is computed by (numerically) integrating the non-Gaussian likelihood.
    # * m.predict_f_samples returns samples of the latent function
    # * m.predict_density returns the log-density of the points Ynew at Xnew.
    xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'C0', lw=2)
    plt.fill_between(xx[:, 0],
                     mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                     mean[:, 0] + 2 * np.sqrt(var[:, 0]),
                     color='C0', alpha=0.2)
    plt.xlim(-0.1, 1.1)
    plt.show()


def mle(X, Y):
    k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    meanf = gpflow.mean_functions.Linear(1.0, 0.0)
    m = gpflow.models.GPR(X, Y, k, meanf)
    m.likelihood.variance = 0.01

    print(m.as_pandas_table())

    gpflow.train.ScipyOptimizer().minimize(m)
    return m

def mcmc(X, Y):
    k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    meanf = gpflow.mean_functions.Linear(1.0, 0.0)
    m = gpflow.models.GPR(X, Y, k, meanf)
    m.clear()

    m.kern.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
    m.kern.variance.prior = gpflow.priors.Gamma(1., 1.)
    m.likelihood.variance.prior = gpflow.priors.Gamma(1., 1.)
    m.mean_function.A.prior = gpflow.priors.Gaussian(0., 10.)
    m.mean_function.b.prior = gpflow.priors.Gaussian(0., 10.)
    m.compile()
    print(m.as_pandas_table())

    sampler = gpflow.train.HMC()
    samples = sampler.sample(m, num_samples=500, burn=200, epsilon=0.05, lmin=10, lmax=20,
                             logprobs=False)
    return samples, m

def eval():
    X, Y = get_data()
    plt.plot(X, Y, 'kx', mew=2)
    plt.show()

    m1 = gpr(X, Y)
    plot(X, Y, m1)
    print(m1.as_pandas_table())
    m1.clear()

    m2 = mle(X, Y)
    plot(X, Y, m2)
    print(m2.as_pandas_table())
    m2.clear()

    samples, m3 = mcmc(X, Y)
    plot(X, Y, m3)
    print(m3.as_pandas_table())

    plt.figure(figsize=(8, 4))
    for i, col in samples.iteritems():
        plt.plot(col, label=col.name)
    plt.legend(loc=0)
    plt.xlabel('hmc iteration')
    plt.ylabel('parameter value')

    f, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].plot(samples['GPR/likelihood/variance'],
                samples['GPR/kern/variance'], 'k.', alpha=0.15)
    axs[0].set_xlabel('noise_variance')
    axs[0].set_ylabel('signal_variance')

    axs[1].plot(samples['GPR/likelihood/variance'],
                samples['GPR/kern/lengthscales'], 'k.', alpha=0.15)
    axs[1].set_xlabel('noise_variance')
    axs[1].set_ylabel('lengthscale')

    axs[2].plot(samples['GPR/kern/lengthscales'],
                samples['GPR/kern/variance'], 'k.', alpha=0.1)
    axs[2].set_xlabel('lengthscale')
    axs[2].set_ylabel('signal_variance')
    plt.show()

    # plot the function posterior
    xx = np.linspace(-0.1, 1.1, 100)[:, None]
    plt.figure(figsize=(12, 6))
    for i, s in samples.iloc[::20].iterrows():
        f = m3.predict_f_samples(xx, 1, initialize=False, feed_dict=m3.sample_feed_dict(s))
        plt.plot(xx, f[0, :, :], 'C0', lw=2, alpha=0.1)

    plt.plot(X, Y, 'kx', mew=2)
    _ = plt.xlim(xx.min(), xx.max())
    _ = plt.ylim(0, 6)
    plt.show()
    m3.clear()