import numpy as np

np.random.seed(1)


def make_data(points=25):
    # f = lambda x: x * np.sin(x)
    def f(x_):
        return np.sin(10 * x_[..., 0]) * np.exp(-x_[..., 0] ** 2)
    X = np.random.uniform(-1., 1., points)[..., np.newaxis]
    Y = np.random.normal(f(X), .05)

    x = np.linspace(-1., 1., 200)[..., np.newaxis]

    return X, Y, x, f
