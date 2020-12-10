#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
run games and save corresponding results in an appropriate file.

Usage:
  play_games.py <nb_trials> <nb_games>
                (--Random
                    | --Oracle
                    | --EtC <m>
                    | --EtFtL <m>
                    | --eGreedy <c>
                    | --UCB [<alpha>]
                    | --TS
                )
                (--Ber <theta> ...)


Options:
  -h --help         Show this screen
"""

import numpy as np
from copy import deepcopy


def play_games(players, arms, nb_trials, nb_games):
    """
    Play one game and return stored informations

    :param players:
    :param arms:
    :param nb_trials:
    :param nb_games:
    :return:
    """
    logs = []
    for i_p, player in enumerate(players):
        # --- To save logs ---
        best_mean = max([a.mean() for a in arms])
        chosen_arm = np.zeros((nb_games, nb_trials))
        reward = np.zeros((nb_games, nb_trials))
        expected_reward = np.zeros((nb_games, nb_trials))
        expected_best_reward = np.zeros((nb_games, nb_trials))

        # --- Play games ---
        for game in range(nb_games):
            player_for_one_game = deepcopy(player)
            player_for_one_game.restart()
            for t in range(nb_trials):
                # play one turn
                i = player_for_one_game.choose_next_arm()
                # print("%d\t%f" % (i, arms[i].mean()))
                rew = arms[i].draw()
                player_for_one_game.update(i, rew)
                # store informations
                chosen_arm[game, t] = i
                reward[game, t] = rew
                expected_reward[game, t] = arms[i].mean()
                expected_best_reward[game, t] = best_mean
        logs.append({'chosen_arm': chosen_arm,
                     'reward': reward,
                     'expected_reward': expected_reward,
                     'expected_best_reward': expected_best_reward,
                     'label': player.label()
                     })

    # --- End ---
    return logs





if __name__ == "__main__":
    from docopt import docopt

    from arm import Bernoulli, Gaussian, TruncatedExponential
    from player import Random, Oracle, ExploreThenCommit, EpsilonNGreedy, UCB1, ThompsonSamplingBernoulli
    from tools import record_zip

    arguments = docopt(__doc__)
    print(arguments)

    file_name = "data/"
    # --- environment ---
    if arguments['--Ber']:
        arms = [Bernoulli(float(p)) for p in arguments['<theta>']]
        file_name += 'Ber' + '_'.join(arguments['<theta>'])
    nb_arms = len(arms)
    # --- player ---
    file_name += '__'
    if arguments['--Random']:
        players = [Random(nb_arms=nb_arms)]
        file_name += 'Random'
    elif arguments['--Oracle']:
        players = [Oracle(np.argmax([arm.mean() for arm in arms]))]
        file_name += 'Oracle'
    elif arguments['--EtC']:
        players = [ExploreThenCommit(nb_arms=nb_arms, n0=int(arguments['<m>']))]
        file_name += 'EtC__m_' + arguments['<m>']
    elif arguments['--eGreedy']:
        players = [EpsilonNGreedy(nb_arms=nb_arms, c=float(arguments['<c>']))]
        file_name += 'eGreedy__c_' + arguments['<c>']
    elif arguments['--UCB']:
        players = [UCB1(nb_arms=nb_arms, alpha=float(arguments['<alpha>']))]
        file_name += 'UCB__alpha_' + arguments['<alpha>']
    elif arguments['--TS']:
        players = [ThompsonSamplingBernoulli(nb_arms=nb_arms)]
        file_name += 'TS'

    # --- play ---
    logs = play_games(players, arms, int(arguments['<nb_trials>']), int(arguments['<nb_games>']))
    file_name += '__nb_trials_' + arguments['<nb_trials>']\
                 + '__nb_games_' + arguments['<nb_games>']

    # --- save ---
    file_name += ".gz"

    record_zip(file_name, logs)
