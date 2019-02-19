import numpy as np
import os

from .data.gen import make_data
import gp.gp_sklearn


def run(output="output/"):
    X, Y, x, f = make_data()

    y, sigma = gp.gp_sklearn.evalMLENoiseless(X, Y, x)
    gp.gp_sklearn.plot(X, Y, x, y, sigma, DY=None, f=f, output=os.path.join(output, "sklearn_mle_noiseless.png"))

    # ----------------------------------------------------------------------
    # now the noisy case

    # Observations and noise
    dy = 0.5
    DY = dy * np.random.random(Y.shape)
    noise = np.random.normal(0, DY)
    Y += noise

    y, sigma = gp.gp_sklearn.evalMLENoisy(X, Y, x, dy)
    gp.gp_sklearn.plot(X, Y, x, y, sigma, DY, f, output=os.path.join(output, "sklearn_mle_noise.png"))
