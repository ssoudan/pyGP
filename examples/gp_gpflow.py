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
import gp.gp_gpflow


def evalHandcrafted(X, Y):
    k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    m = gpflow.models.GPR(X, Y, kern=k)
    m.likelihood.variance = 0.01

    return m


if __name__ == '__main__':
    X, Y, x, f = data.make_data()
    Y = np.atleast_2d(Y).T

    plt.plot(X, Y, 'kx', mew=2)
    plt.show()

    m1 = evalHandcrafted(X, Y)
    gp.gp_gpflow.plot(X, Y, x, m1, 'handcrafted GP model', f)
    print(m1.as_pandas_table())
    m1.clear()

    _, m2 = gp.gp_gpflow.evalMLE(X, Y)
    gp.gp_gpflow.plot(X, Y, x, m2, 'MLE-fitted model', f)
    print(m2.as_pandas_table())
    m2.clear()

    traces, m3 = gp.gp_gpflow.evalMCMC(X, Y)
    gp.gp_gpflow.plot(X, Y, x, m3, 'MCMC-fitted model', f)
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

