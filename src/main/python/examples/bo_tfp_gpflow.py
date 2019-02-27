import os
import numpy as np
import warnings
import matplotlib.pylab as plt
import matplotlib.lines as mlines
import seaborn as sns
from .data.gen import make_data
from opt.bo import Optimizer
from gp.gp_tfp import evalMLE, evalHMC, plot
from gp.gp_gpflow import evalMCMCSamples, evalMLESamples

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def eval(w, m, output, X, Y, x, f, f_noisy, steps=10):
    if m is "mle":
        if w is "tfp":
            opt = Optimizer(evalMLE)
        else:
            opt = Optimizer(evalMLESamples)
    else:
        if w is "tfp":
            opt = Optimizer(evalHMC)
        else:
            opt = Optimizer(evalMCMCSamples)

    for i, xx in enumerate(X):
        opt.add_point(float(xx), float(Y[i]))

    print("----- RUNNING %s %s -----" % (w, m))

    best = []

    for i in range(steps):
        x_star, y, acquisition = opt.find_next_candidate()
        print("x_star=", x_star)

        X_, Y_ = opt.get_points()
        plot(X_, Y_, x, y,
             f=f,
             acquisition=acquisition,
             next_X=x_star,
             title="%s %s" % (w, m),
             output=os.path.join(output, "opt_%s_%s_%d.png" % (w, m, i)))

        y_star = f_noisy(x_star)
        print("y_star=", y_star)
        opt.add_point(float(x_star), float(y_star))

        best.append(np.max(np.append(Y_, y_star)))

    print("x_star=", x_star)

    return best


def load_and_plot(output="output/opt"):

    f = os.path.join(output, 'opt_gpflow_tfp_hmc_v_mle.npz')
    d = np.load(f)

    y_stars_gpflow_mles_ = d[d.files[0]]
    y_stars_gpflow_hmcs_ = d[d.files[1]]
    y_stars_tfp_mles_ = d[d.files[2]]
    y_stars_tfp_hmcs_ = d[d.files[3]]

    plt.figure()
    with sns.axes_style("white"):
        a = y_stars_gpflow_mles_
        plt.plot(np.array([np.arange(a.shape[1])] * a.shape[0]).T, a.T, color="k", alpha=0.8)
        plt.fill_between(np.arange(a.shape[1]), np.min(a, axis=0), np.max(a, axis=0), color="k", alpha=0.1)

        a = y_stars_gpflow_hmcs_
        plt.plot(np.array([np.arange(a.shape[1])] * a.shape[0]).T, a.T, color="r", alpha=0.8)
        plt.fill_between(np.arange(a.shape[1]), np.min(a, axis=0), np.max(a, axis=0), color="r", alpha=0.1)

        a = y_stars_tfp_mles_
        plt.plot(np.array([np.arange(a.shape[1])] * a.shape[0]).T, a.T, color="c", alpha=0.8)
        plt.fill_between(np.arange(a.shape[1]), np.min(a, axis=0), np.max(a, axis=0), color="c", alpha=0.1)

        a = y_stars_tfp_hmcs_
        plt.plot(np.array([np.arange(a.shape[1])] * a.shape[0]).T, a.T, color="m", alpha=0.8)
        plt.fill_between(np.arange(a.shape[1]), np.min(a, axis=0), np.max(a, axis=0), color="m", alpha=0.1)

    l1 = mlines.Line2D([], [], color='k', label='gpflow mle')
    l2 = mlines.Line2D([], [], color='r', label='gpflow mcmc')
    l3 = mlines.Line2D([], [], color='c', label='tfp mle')
    l4 = mlines.Line2D([], [], color='m', label='tfp mcmc')

    plt.legend(handles=[l1, l2, l3, l4])

    plt.show()
    plt.close()


def run(output="output/opt/"):
    y_stars_gpflow_mles = []
    y_stars_gpflow_mcmcs = []
    y_stars_tfp_mles = []
    y_stars_tfp_mcmcs = []

    steps = 20
    rep = 5

    for i in range(rep):
        output_rep = os.path.join(output, "rep%d" % i)
        os.makedirs(output_rep, exist_ok=True)

        print("========== %d =========" % i)
        X, Y, x, f, f_noisy = make_data(4)

        y_stars_gpflow_mle = eval("gpflow", "mle", output_rep, X, Y, x, f, f_noisy, steps=steps)
        y_stars_gpflow_mcmc = eval("gpflow", "mcmc", output_rep, X, Y, x, f, f_noisy, steps=steps)

        y_stars_tfp_mle = eval("tfp", "mle", output_rep, X, Y, x, f, f_noisy, steps=steps)
        y_stars_tfp_mcmc = eval("tfp", "mcmc", output_rep, X, Y, x, f, f_noisy, steps=steps)

        y_stars_gpflow_mles.append(y_stars_gpflow_mle)
        y_stars_gpflow_mcmcs.append(y_stars_gpflow_mcmc)
        y_stars_tfp_mles.append(y_stars_tfp_mle)
        y_stars_tfp_mcmcs.append(y_stars_tfp_mcmc)

    plt.figure(figsize=(12, 6))
    for y_stars in y_stars_gpflow_mles:
        plt.step(np.arange(len(y_stars)), y_stars, 'r+-', where='post', alpha=0.5)

    for y_stars in y_stars_gpflow_mcmcs:
        plt.step(np.arange(len(y_stars)), y_stars, 'c+-', where='post', alpha=0.5)

    for y_stars in y_stars_tfp_mles:
        plt.step(np.arange(len(y_stars)), y_stars, 'm+-', where='post', alpha=0.5)

    for y_stars in y_stars_tfp_mcmcs:
        plt.step(np.arange(len(y_stars)), y_stars, 'g+-', where='post', alpha=0.5)

    plt.title("mle vs hmc/tfp vs gpflow bo")

    l1 = mlines.Line2D([], [], color='r', label='gpflow mle')
    l2 = mlines.Line2D([], [], color='c', label='gpflow mcmc')
    l3 = mlines.Line2D([], [], color='m', label='tfp mle')
    l4 = mlines.Line2D([], [], color='g', label='tfp mcmc')

    plt.legend(handles=[l1, l2, l3, l4])
    plt.savefig(os.path.join(output, "opt_gpflow_tfp_hmc_v_mle.png"))
    plt.show()
    plt.close()

    y_stars_gpflow_mles_ = np.array(y_stars_gpflow_mles)
    y_stars_gpflow_hmcs_ = np.array(y_stars_gpflow_mcmcs)
    y_stars_tfp_mles_ = np.array(y_stars_tfp_mles)
    y_stars_tfp_hmcs_ = np.array(y_stars_tfp_mcmcs)
    np.savez(os.path.join(output, "opt_gpflow_tfp_hmc_v_mle.npz"),
             y_stars_gpflow_mles_=y_stars_gpflow_mles_,
             y_stars_gpflow_hmcs_=y_stars_gpflow_hmcs_,
             y_stars_tfp_mles_=y_stars_tfp_mles_,
             y_stars_tfp_hmcs_=y_stars_tfp_hmcs_)
