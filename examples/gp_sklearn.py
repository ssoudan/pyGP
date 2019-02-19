import numpy as np

import examples.data as data
import gp.gp_sklearn


if __name__ == '__main__':
    X, Y, x, f = data.make_data()

    y, sigma = gp.gp_sklearn.evalMLENoiseless(X, Y, x)
    gp.gp_sklearn.plot(X, Y, x, y, sigma, DY=None, f=f)

    # ----------------------------------------------------------------------
    # now the noisy case

    # Observations and noise
    dy = 0.5
    DY = dy * np.random.random(Y.shape)
    noise = np.random.normal(0, DY)
    Y += noise

    y, sigma = gp.gp_sklearn.evalMLENoisy(X, Y, x, dy)
    gp.gp_sklearn.plot(X, Y, x, y, sigma, DY, f)
