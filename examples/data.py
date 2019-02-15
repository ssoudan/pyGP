import numpy as np

np.random.seed(1)

def make_data():
    # f = lambda x: x * np.sin(x)
    f = lambda x: np.sin(10 * x[..., 0]) * np.exp(-x[..., 0] ** 2)
    X = np.random.uniform(-1., 1., 25)[..., np.newaxis]
    Y = np.random.normal(f(X), .05)

    x = np.linspace(-1., 1., 200)[..., np.newaxis]

    return X, Y, x, f
