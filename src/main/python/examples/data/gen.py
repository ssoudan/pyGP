import numpy as np

np.random.seed(1)

noise = 0.25

def make_data(points=25):
    # f = lambda x: x * np.sin(x)
    def f(x_):
        return np.sin(10 * x_[..., 0]) * np.exp(-x_[..., 0] ** 2)
    X = np.random.uniform(-1., 1., points)[..., np.newaxis]
    Y = np.random.normal(f(X), noise)

    x = np.linspace(-1., 1., 200)[..., np.newaxis]

    def f_noisy(x_):
        return np.random.normal(f(x_), noise)

    return X, Y, x, f, f_noisy
