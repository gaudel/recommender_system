# Run games and print corresponding curves
# ===========================

import numpy as np
import matplotlib.pyplot as plt
from math import floor
from play_games import play_games


# =============================
# Tools
# =============================

def cumulative_reward(logs):
    """
    compute average cumulative reward
    :param logs:
    :return:
    """
    return np.mean(np.cumsum(logs['reward'], axis=1), axis=0)


def cumulative_regret(logs):
    """
    compute average cumulative regret
    :param logs:
    :return:
    """
    return np.mean(np.cumsum(logs['expected_best_reward'] - logs['reward'], axis=1), axis=0)


def nb_times_best__dist(logs):
    """
    compute average cumulative regret
    :param logs:
    :return:
    """
    return np.sum(logs['expected_best_reward'] == logs['expected_reward'], axis=1)


def logarithmic_indices(stop, n):
    """
    returns n indices logarithmically spanned from 0 to stop-1
    :param stop:
    :param n:
    :return:
    """
    return np.unique([floor(np.exp(i / (n - 1) * np.log(stop))) - 1 for i in range(n)])


# =============================
# Plots
# =============================

def print_cum_rew(logs, T=None):
    if logs[0]['reward'].shape[0] == 1:
        print("Cumulative Reward at time-stamp %d" % (logs[0]['reward'].shape[1]))
        for log in logs:
            print("%15s     %0.2f" % (log['label'], np.sum(log['reward'][:, :T], axis=1)[0]))
    else:
        print("Cumulative Reward at time-stamp %d (in average)" % (logs[0]['reward'].shape[1]))
        for log in logs:
            vals = np.sum(log['reward'][:, :T], axis=1)
            print(
                "%15s     mean: %0.2f\t min: %0.2f\t max: %0.2f" % (log['label'], vals.mean(), vals.min(), vals.max()))


def plot_cum_reg(logs, subplot=False, T=None):
    if subplot:
        plt.subplot(1, 2, 1)
    else:
        plt.clf()
    inds = logarithmic_indices(logs[0]['reward'][:, :T].shape[1], 100)
    for i_p, log in enumerate(logs):
        plt.plot(inds + 1, cumulative_regret(log)[inds], label=log['label'], color='C' + str(i_p % 10))
    plt.xlabel('Time')
    plt.ylabel('Average Cumulative Regret')
    plt.legend()
    plt.grid(True)
    if not subplot:
        plt.show()


def hist_cum_reg(logs, subplot=False, logscale=False, xlim=None, T=None):
    # --- Compute common slices ---
    v_min = np.inf
    v_max = -np.inf
    for log in logs:
        v_min = min(v_min, np.sum(log['reward'][:, :T], axis=1).min())
        v_max = max(v_min, np.sum(log['reward'][:, :T], axis=1).max())
    nb_bins = min(max(logs[0]['reward'][:, :T].shape[0] // 5, 10), 100)
    bins = np.linspace(v_min, v_max + 0.00000001, nb_bins)

    # --- Draw histogram ---
    if subplot:
        plt.subplot(1, 2, 2)
    else:
        plt.clf()
    for i_p, log in enumerate(logs):
        plt.hist(np.sum(log['reward'][:, :T], axis=1), bins=bins, density=False, label=log['label'],
                 facecolor='C' + str(i_p % 10), alpha=0.75, log=logscale)
    plt.xlabel('Cumulative Reward')
    plt.ylabel('Frequency')
    plt.legend()
    if not xlim is None:
        plt.xlim(xlim)
    plt.grid(True)
    if not subplot:
        plt.show()


def plot_exp(logs, T=None, hist_xlim=None, hist_logscale=False):
    """
    Plot results as in the Notebook

    :param logs:
    :return:
    """
    print_cum_rew(logs, T=T)
    if logs[0]['reward'].shape[0] == 1:
        plot_cum_reg(logs, T=T)
    else:
        plt.clf()
        plt.figure(figsize=(13, 4.8))
        plot_cum_reg(logs, subplot=True, T=T)
        hist_cum_reg(logs, subplot=True, logscale=hist_logscale, xlim=hist_xlim, T=T)
        plt.show()


# =============================
# play_games
# =============================

def run_exp(players, arms, nb_trials, nb_games):
    """
    Play one game and return stored information

    :param players:
    :param arms:
    :param nb_trials:
    :param nb_games:
    :return:
    """
    plot_exp(play_games(players, arms, nb_trials, nb_games))


if __name__ == '__main__':
    pass
