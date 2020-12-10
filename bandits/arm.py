# List of potential arms
# ===========================

from random import betavariate
from numpy.random import random, normal
from math import exp, log


class MyBeta:
    """
    beta law with parameters alpha and beta such that $E_{X\sim law}[X] = mean$
    """
    def __init__(self, mean):
        assert(mean > 0)
        assert(mean < 1)
        self.alpha = mean
        self.beta = 1-mean

    def draw(self):
        return betavariate(self.alpha, self.beta)

    def mean(self):
        return self.alpha

    def __str__(self):
        return str(self.mean)

class Bernoulli:
    """
    returns 1 with probability p and 0 otherwise
    """
    def __init__(self, mean):
        pass # XXX TO DO XXX

    def draw(self):
        return 0 # XXX TO DO XXX

    def mean(self):
        return 0. # XXX TO DO XXX

    def __str__(self):
        return "" # XXX TO DO XXX


class Gaussian:
    """
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def draw(self):
        return normal(self.mu, self.sigma)

    def mean(self):
        return self.mu

    def stddev(self):
        return self.sigma

    def __str__(self):
        return str(self.mu)


class TruncatedExponential:
    """
    """

    def __init__(self, p, trunc):
        self.p = p
        self.trunc = trunc
        self.mean = (1. - exp(-p * trunc)) / p

    def draw(self):
        return min(-(1 / self.p) * log(random()), self.trunc)

    def mean(self):
        return (1. - exp(-self.p * self.trunc)) / self.p

    def __str__(self):
        return str(self.p)



