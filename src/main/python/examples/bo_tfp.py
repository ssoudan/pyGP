import os
import numpy as np
import matplotlib.pylab as plt
from .data.gen import make_data
from opt.bo import Optimizer
from gp.gp_tfp import evalMLE, evalHMC, plot


def eval(m, output, X, Y, x, f):
    if m is "MLE":
        opt = Optimizer(evalMLE)
    else:
        opt = Optimizer(evalHMC)

    for i, xx in enumerate(X):
        opt.add_point(float(xx), float(Y[i]))

    best = []

    for i in range(10):
        x_star = opt.find_next_candidate()
        print("x_star=", x_star)

        X_, Y_ = opt.get_points()
        plot(X_, Y_, x, None, f, title=m, output=os.path.join(output, "opt_tfp_%s_%d.png" % (m, i)))

        y_star = f(x_star)
        print("y_star=", y_star)
        opt.add_point(float(x_star), float(y_star))

        best.append(np.max(np.append(Y_, y_star)))

    print("x_star=", x_star)

    return best


def mle(output, X, Y, x, f):
    return eval("MLE", output, X, Y, x, f)


def hmc(output, X, Y, x, f):
    return eval("HMC", output, X, Y, x, f)


def run(output="output/"):
    y_stars_mles = []
    y_stars_hmcs = []

    for i in range(10):
        print("========== %d =========" % i)
        X, Y, x, f = make_data(4)

        y_stars_mle = mle(output, X, Y, x, f)
        y_stars_hmc = hmc(output, X, Y, x, f)

        y_stars_hmcs.append(y_stars_hmc)
        y_stars_mles.append(y_stars_mle)

    plt.figure(figsize=(12, 6))
    for y_stars in y_stars_mles:
        l1 = plt.step(np.arange(len(y_stars)), y_stars, 'r+-', where='post', alpha=0.5)

    for y_stars in y_stars_hmcs:
        l2 = plt.step(np.arange(len(y_stars)), y_stars, 'c+-', where='post', alpha=0.5)

    plt.title("mle vs hmc bo")
    plt.legend((l1, l2), ('mle', 'hmc'))
    plt.savefig(os.path.join(output, "opt_tfp_hmc_v_mle.png"))
    plt.show()
    plt.close()

    y_stars_mles_ = np.array(y_stars_mles)
    y_stars_hmcs_ = np.array(y_stars_hmcs)
    np.savez(os.path.join(output, "opt_tfp_hmc_v_mle.npz"), y_stars_mles_, y_stars_hmcs_)
