# List potential players
# ===========================

from random import randint, random
import numpy as np
from numpy.random import beta
from random import random


class Random:
    """
    Player which plays arms at random
    """

    def __init__(self, nb_arms):
        self.nb_arms = nb_arms

    def choose_next_arm(self):
        return randint(0, self.nb_arms - 1)

    def update(self, arm, reward):
        pass

    def restart(self):
        pass

    def label(self):
        return "Random"


class Oracle:
    """
    Player which plays the best arm
    """

    def __init__(self, best_arm):
        self.best_arm = best_arm

    def choose_next_arm(self):
        return self.best_arm

    def update(self, arm, reward):
        pass

    def restart(self):
        pass

    def label(self):
        return "Oracle"


class ExploreThenCommit:
    """
    Player which plays arms at random for n0 trials, and then plays the best arm up to now
    """

    def __init__(self, nb_arms, n0):
        self.cum_reward = np.zeros(nb_arms)
        self.nb_trials = np.zeros(nb_arms, dtype=np.uint)
        self.n0 = n0
        self.winner = -1

    def choose_next_arm(self):
        t = np.sum(self.nb_trials)
        if t < self.n0:
            # explore
            return (np.uint(t % self.cum_reward.shape[0]))
        else:
            # exploit
            return self.winner

    def update(self, arm, reward):
        self.cum_reward[arm] += reward
        self.nb_trials[arm] += 1
        T = np.sum(self.nb_trials)
        if T == self.n0:
            self.winner = np.argmax(self.cum_reward / self.nb_trials)

    def restart(self):
        nb_arms = self.cum_reward.shape[0]
        self.cum_reward = np.zeros(nb_arms)
        self.nb_trials = np.zeros(nb_arms, dtype=np.uint)
        self.winner = -1

    def label(self):
        return "EtC, n0=%d" % (self.n0)


class EpsilonNGreedy:
    """
    Player which plays the best arm (up to now) with probability (1-c/t), and an arm uniformly at random otherwise
    """

    def __init__(self, nb_arms, c):
        self.cum_reward = np.zeros(nb_arms)
        self.nb_trials = np.zeros(nb_arms, dtype=np.uint)
        self.c = c

    def choose_next_arm(self, epsilon=10 ** (-5)):
        t = sum(self.nb_trials) + epsilon
        if random() < self.c / t:
            # explore
            return randint(0, self.cum_reward.shape[0] - 1)
        else:
            # exploit
            return np.argmax(self.cum_reward / (self.nb_trials + epsilon))

    def update(self, arm, reward):
        self.cum_reward[arm] += reward
        self.nb_trials[arm] += 1

    def restart(self):
        nb_arms = self.cum_reward.shape[0]
        self.cum_reward = np.zeros(nb_arms)
        self.nb_trials = np.zeros(nb_arms, dtype=np.uint)

    def label(self):
        return "$\\epsilon_n$-Greedy, c=%.3f" % (self.c)


class UCB1:
    """
    Player which plays the arm with the highest confidence upper confidence bound
    """

    def __init__(self, nb_arms, alpha=1.):
        self.alpha = alpha
        pass # XXX TO DO XXX

    def choose_next_arm(self, epsilon=10**(-5)):
        return 0 # XXX TO DO XXX

    def update(self, arm, reward):
        pass # XXX TO DO XXX

    def restart(self):
        pass # XXX TO DO XXX

    def label(self):
        return "UCB, $\\alpha$=%.3f" % (self.alpha)



class ThompsonSamplingBernoulli:
    """
    Approximate random sampling given posterior probability to be optimal
    """

    def __init__(self, nb_arms, prior_s=0.5, prior_f=0.5):
        self.prior_s = prior_s
        self.prior_f = prior_f
        self.success = np.ones(nb_arms, dtype=np.uint) * prior_s
        self.failure = np.ones(nb_arms, dtype=np.uint) * prior_f

    def choose_next_arm(self):
        thetas = beta(self.success, self.failure)
        return np.argmax(thetas)

    def update(self, arm, reward):
        rew = int(random() < reward)
        self.success[arm] += rew
        self.failure[arm] += 1 - rew

    def restart(self):
        nb_arms = self.success.shape[0]
        self.success = np.ones(nb_arms, dtype=np.uint)*self.prior_s
        self.failure = np.ones(nb_arms, dtype=np.uint)*self.prior_f

    def label(self):
        return "TS"

