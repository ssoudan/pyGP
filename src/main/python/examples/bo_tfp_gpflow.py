import os
import numpy as np
import warnings
import matplotlib.pylab as plt
import matplotlib.lines as mlines
from .data.gen import make_data
from opt.bo import Optimizer
from gp.gp_tfp import evalMLE, evalHMC, plot
from gp.gp_gpflow import evalMCMCSamples, evalMLESamples

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def eval(w, m, output, X, Y, x, f, steps=10):
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
        x_star, y = opt.find_next_candidate()
        print("x_star=", x_star)

        X_, Y_ = opt.get_points()
        plot(X_, Y_, x, y, f, title="%s %s" % (w, m), output=os.path.join(output, "opt_%s_%s_%d.png" % (w, m, i)))

        y_star = f(x_star)
        print("y_star=", y_star)
        opt.add_point(float(x_star), float(y_star))

        best.append(np.max(np.append(Y_, y_star)))

    print("x_star=", x_star)

    return best


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
        X, Y, x, f = make_data(4)

        y_stars_gpflow_mle = eval("gpflow", "mle", output_rep, X, Y, x, f, steps=steps)
        y_stars_gpflow_mcmc = eval("gpflow", "mcmc", output_rep, X, Y, x, f, steps=steps)

        y_stars_tfp_mle = eval("tfp", "mle", output_rep, X, Y, x, f, steps=steps)
        y_stars_tfp_mcmc = eval("tfp", "mcmc", output_rep, X, Y, x, f, steps=steps)

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
             y_stars_gpflow_mles_, y_stars_gpflow_hmcs_,
             y_stars_tfp_mles_, y_stars_tfp_hmcs_)
