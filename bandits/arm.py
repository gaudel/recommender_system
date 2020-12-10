# List of potential arms
# ===========================

from random import random, betavariate


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
        return str(self.p)

class Bernoulli:
    """
    returns 1 with probability p and 0 otherwise
    """
    def __init__(self, p):
        pass # XXX TO DO XXX

    def draw(self):
        return 0 # XXX TO DO XXX

    def mean(self):
        return 0. # XXX TO DO XXX

    def __str__(self):
        return "" # XXX TO DO XXX


