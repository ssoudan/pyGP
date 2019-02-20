from scipy.stats import norm

eps = 1e-12


def ExpectedImprovement(tau, mean, std):
    """
    Expected Improvement acquisition function.
    Parameters
    ----------
    tau: float
        Best observed function evaluation.
    mean: float
        Point mean of the posterior process.
    std: float
        Point std of the posterior process.
    Returns
    -------
    float
        Expected improvement.
    """
    z = (mean - tau - eps) / (std + eps)
    r = (mean - tau) * norm.cdf(z) + std * norm.pdf(z)
    r[std == 0] = 0
    return r


def ProbabilityImprovement(tau, mean, std):
    """
    Probability of Improvement acquisition function.

    Parameters
    ----------
    tau: float
        Best observed function evaluation.
    mean: float
        Point mean of the posterior process.
    std: float
        Point std of the posterior process.

    Returns
    -------
    float
        Probability of improvement.
    """
    z = (mean - tau - eps) / (std + eps)
    return norm.cdf(z)


def UCB(tau, mean, std, beta=1.5):
    return mean + beta * std
