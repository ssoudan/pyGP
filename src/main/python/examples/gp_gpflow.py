import gpflow
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os
from .data.gen import make_data
import gp.gp_gpflow
import seaborn as sns


matplotlib.rcParams['figure.figsize'] = (12, 6)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def evalHandcrafted(X, Y):
    k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    m = gpflow.models.GPR(X, Y, kern=k)
    m.likelihood.variance = 0.01

    return m


def run(output="output/"):
    X, Y, x, f = make_data()
    Y = np.atleast_2d(Y).T

    plt.plot(X, Y, 'kx', mew=2)
    plt.savefig(os.path.join(output, "gpflow_input_data.png"))
    plt.show()

    m1 = evalHandcrafted(X, Y)
    gp.gp_gpflow.plot(X, Y, x, m1, 'handcrafted GP model', f,
                      output=os.path.join(output, "gpflow_handcrafted_model.png"))
    print(m1.as_pandas_table())
    m1.clear()

    _, m2 = gp.gp_gpflow.evalMLE(X, Y)
    gp.gp_gpflow.plot(X, Y, x, m2, 'MLE-fitted model', f, output=os.path.join(output, "gpflow_mle.png"))
    print(m2.as_pandas_table())

    # plot the function posterior
    plt.figure(figsize=(12, 6))
    num_samples = 10
    ff = m2.predict_f_samples(x, num_samples, initialize=False)
    plt.plot(np.stack([x[:, 0]] * num_samples).T, ff[:, :, 0].T, 'C0', lw=2, alpha=0.1)
    plt.plot(X, Y, 'kx', mew=2)
    _ = plt.xlim(x.min(), x.max())
    plt.title('Posterior samples - MLE')
    plt.savefig(os.path.join(output, "gpflow_mle_posterior_samples.png"))
    plt.show()
    m2.clear()

    traces, m3 = gp.gp_gpflow.evalMCMC(X, Y)
    gp.gp_gpflow.plot(X, Y, x, m3, 'MCMC-fitted model', f, output=os.path.join(output, "gpflow_mcmc.png"))
    print(m3.as_pandas_table())

    plt.figure(figsize=(8, 4))
    for i, col in traces.iteritems():
        plt.plot(col, label=col.name)
    plt.legend(loc=0)
    plt.xlabel('HMC iteration')
    plt.ylabel('parameter value')
    plt.title('HMC traces')
    plt.savefig(os.path.join(output, "gpflow_mcmc_traces.png"))
    plt.show()

    fig = plt.figure(figsize=(12, 4))
    axs0 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1, fig=fig)

    axs0.plot(traces['GPR/likelihood/variance'],
              traces['GPR/kern/variance'], 'k.', alpha=0.15)
    axs0.set_xlabel('noise_variance')
    axs0.set_ylabel('signal_variance')

    axs01 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=1, fig=fig)
    sns.distplot(traces['GPR/likelihood/variance'], color='m', ax=axs01)
    axs01.set_xlim(axs0.get_xlim())
    plt.setp(axs01, yticks=[])

    axs1 = plt.subplot2grid((3, 3), (0, 1), rowspan=2, colspan=1, fig=fig)

    axs1.plot(traces['GPR/kern/lengthscales'],
              traces['GPR/likelihood/variance'], 'k.', alpha=0.15)
    axs1.set_xlabel('lengthscale')
    axs1.set_ylabel('noise_variance')

    axs11 = plt.subplot2grid((3, 3), (2, 1), rowspan=1, colspan=1, fig=fig)
    sns.distplot(traces['GPR/kern/lengthscales'], color='m', ax=axs11)
    axs11.set_xlim(axs1.get_xlim())
    plt.setp(axs11, yticks=[])

    axs2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2, colspan=1, fig=fig)

    axs2.plot(traces['GPR/kern/variance'],
              traces['GPR/kern/lengthscales'], 'k.', alpha=0.1)
    axs2.set_xlabel('signal_variance')
    axs2.set_ylabel('lengthscale')

    axs21 = plt.subplot2grid((3, 3), (2, 2), rowspan=1, colspan=1, fig=fig)
    sns.distplot(traces['GPR/kern/variance'], color='m', ax=axs21)
    axs21.set_xlim(axs2.get_xlim())
    plt.setp(axs21, yticks=[])

    fig.suptitle('HMC (joint) distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output, "gpflow_mcmc_joint_distribution.png"))
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
    plt.savefig(os.path.join(output, "gpflow_mcmc_posterior_samples.png"))
    plt.show()
    m3.clear()
