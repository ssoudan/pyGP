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

import examples.data as data

# def get_data():
#     N = 12
#     X = np.random.rand(N, 1)
#     Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + np.random.randn(N, 1) * 0.1 + 3
#
#     return X, Y

def evalHandcrafted(X, Y):
    k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    m = gpflow.models.GPR(X, Y, kern=k)
    m.likelihood.variance = 0.01

    return m

def plot(X, Y, x, m, t, f = None):

    # GPflow models have several prediction methods:
    # * m.predict_f returns the mean and variance of the latent function (f) at the points Xnew.
    # * m.predict_f_full_cov additionally returns the full covariance matrix of the prediction.
    # * m.predict_y returns the mean and variance of a new data point (i.e. includes the noise varaince). In the case of non-Gaussian likelihoods, the variance is computed by (numerically) integrating the non-Gaussian likelihood.
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
    plt.show()


def evalMLE(X, Y):
    k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    meanf = gpflow.mean_functions.Linear(1.0, 0.0)
    m = gpflow.models.GPR(X, Y, k, meanf)
    m.likelihood.variance = 0.01

    print(m.as_pandas_table())

    gpflow.train.ScipyOptimizer().minimize(m)
    return m

def evalMCMC(X, Y):
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
    samples = sampler.sample(m, num_samples=2000, burn=1000, epsilon=0.05, lmin=1, lmax=20,
                             logprobs=False)
    return samples, m


if __name__ == '__main__':
    X, Y, x, f = data.make_data()
    Y = np.atleast_2d(Y).T

    plt.plot(X, Y, 'kx', mew=2)
    plt.show()

    m1 = evalHandcrafted(X, Y)
    plot(X, Y, x, m1, 'handcrafted GP model', f)
    print(m1.as_pandas_table())
    m1.clear()

    m2 = evalMLE(X, Y)
    plot(X, Y, x, m2, 'MLE-fitted model', f)
    print(m2.as_pandas_table())
    m2.clear()

    traces, m3 = evalMCMC(X, Y)
    plot(X, Y, x, m3, 'MCMC-fitted model', f)
    print(m3.as_pandas_table())

    plt.figure(figsize=(8, 4))
    for i, col in traces.iteritems():
        plt.plot(col, label=col.name)
    plt.legend(loc=0)
    plt.xlabel('HMC iteration')
    plt.ylabel('parameter value')
    plt.title('HMC traces')
    plt.show()

    _, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].plot(traces['GPR/likelihood/variance'],
                traces['GPR/kern/variance'], 'k.', alpha=0.15)
    axs[0].set_xlabel('noise_variance')
    axs[0].set_ylabel('signal_variance')

    axs[1].plot(traces['GPR/likelihood/variance'],
                traces['GPR/kern/lengthscales'], 'k.', alpha=0.15)
    axs[1].set_xlabel('noise_variance')
    axs[1].set_ylabel('lengthscale')

    axs[2].plot(traces['GPR/kern/lengthscales'],
                traces['GPR/kern/variance'], 'k.', alpha=0.1)
    axs[2].set_xlabel('lengthscale')
    axs[2].set_ylabel('signal_variance')
    plt.title('HMC (joint) distribution')
    plt.show()

    # plot the function posterior
    plt.figure(figsize=(12, 6))
    for i, s in traces.iloc[::20].iterrows():
        f = m3.predict_f_samples(x, 1, initialize=False, feed_dict=m3.sample_feed_dict(s))
        plt.plot(x, f[0, :, :], 'C0', lw=2, alpha=0.1)

    plt.plot(X, Y, 'kx', mew=2)
    _ = plt.xlim(x.min(), x.max())
    # _ = plt.ylim(0, 6)
    plt.title('Posterior samples - MCMC')
    plt.show()
    m3.clear()

