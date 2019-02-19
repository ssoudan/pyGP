import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

#
# See https://scikit-learn.org/stable/modules/gaussian_process.html
# See https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
#


def evalMLENoiseless(X, Y, x):  # type: (Any, Any, Any) -> Tuple[Any, Any]

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-4, 1e4)) * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, Y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y, sigma = gp.predict(x, return_std=True)

    return y, sigma


def evalMLENoisy(X, Y, x, DY = 0):   # type: (Any, Any, Any, int) -> Tuple[Any, Any]

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-4, 1e4)) * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=DY ** 2,
                                  n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, Y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y, sigma = gp.predict(x, return_std=True)

    return y, sigma


def plot(X, Y, x, y, sigma, DY=None, f=None, output=None):
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    plt.figure()
    if f is not None:
        plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.errorbar(X.ravel(), Y, DY, fmt='r.', markersize=10, label=u'Observations')
    plt.plot(x, y, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y - 1.9600 * sigma,
                            (y + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    if output is not None:
        plt.savefig(output)
    plt.show()
