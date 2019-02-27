import numpy as np
from acquisition.functions import ExpectedImprovement


class Optimizer(object):

    def __init__(self, eval):
        self.eval = eval

        self.points = set([])
        self.best = None
        self.gp = None

        pass

    def add_point(self, x, y):
        self.points.add((x, y))
        pass

    def find_next_candidate(self):

        if len(self.points) is 0:
            raise Exception("no points")

        x = np.linspace(-1., 1., 200)[..., np.newaxis]

        X_, Y_ = self.get_points()
        _, y = self.eval(X_, Y_, x)

        tau = np.max(Y_)
        mean = np.mean(y, axis=0)
        std = np.std(y, axis=0)

        # print("tau=", tau)
        # print("mean=", mean)
        # print("std=", std)

        ei = ExpectedImprovement(tau, mean, std)

        x_star_idx = np.argmax(ei)
        # print("x_star_idx=", x_star_idx)
        x_star = x[x_star_idx]

        return x_star, y, ei

    def get_points(self):
        X = []
        Y = []
        for (xx, yy) in self.points:
            X.append(xx)
            Y.append(yy)

        X_ = np.atleast_2d(np.array(X)).T
        Y_ = np.array(Y)

        return X_, Y_
