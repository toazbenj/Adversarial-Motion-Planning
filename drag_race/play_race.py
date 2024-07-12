"""
Play Drag Race

Run simulation of race using offline calculated policies, generate experimental data. Run this second after build_race.py.
It will load the saved_game.npz file and pair up the players for several races. Next it will tally the game values of each
run and save them in results.npz. After this you should run plot_race.py to visualize the data.
"""

from graphics import plot_race, plot_pareto_front, plot_average_cost
from utilities import *
import random
import os
import glob

def play_game(policy1, policy2, dynamics, stage_count, init_state_index):
    control1 = np.zeros(stage_count + 1, dtype=int)
    control2 = np.zeros(stage_count + 1, dtype=int)
    states_played = np.zeros(stage_count + 2, dtype=int)
    states_played[0] = init_state_index

    for stage in range(stage_count + 1):
        pick1 = random.random()
        pick2 = random.random()
        current_policy1 = policy1[stage][states_played[stage]]
        current_policy2 = policy2[stage][states_played[stage]]

        total = 0
        for i in range(len(current_policy1)):
            total += current_policy1[i]
            if total > pick1:
                control1[stage] = i
                break

        total = 0
        for i in range(len(current_policy2)):
            total += current_policy2[i]
            if total > pick2:
                control2[stage] = i
                break

        states_played[stage + 1] = dynamics[states_played[stage], control1[stage], control2[stage]]

    return control1, control2, states_played


def find_values(states_played, u, d, rank_cost1, rank_cost2, safety_cost1, safety_cost2):
    """
    Sum up costs of each players' actions
    :param states_played: list of numpy arrays of states
    :param u: player 1 control inputs for each stage
    :param d: player 2 control inputs for each stage
    :param costs1: player 1 costs for all configurations
    :param costs2: player 2 costs for all configurations
    :return: game values for each player list
    """
    values = np.zeros((2, 2))
    for idx in range(len(states_played) - 1):
        round_rank_cost1 = rank_cost1[states_played[idx], u[idx], d[idx]]
        round_rank_cost2 = rank_cost2[states_played[idx], u[idx], d[idx]]

        round_safety_cost1 = safety_cost1[states_played[idx], u[idx], d[idx]]
        round_safety_cost2 = safety_cost2[states_played[idx], u[idx], d[idx]]

        values[0] += np.array([int(round_rank_cost1), int(round_safety_cost1)])
        values[1] += np.array([int(round_rank_cost2), int(round_safety_cost2)])
    return values


def calculate_theoretical_values(safety_cost1, safety_cost2, rank_cost1, rank_cost2, player_pairs):
    theoretical_values = np.zeros((len(player_pairs), 4, ))
    for policies in player_pairs:
        y = policies[0]
        z = policies[1]

        p1_safety_cost = y @ safety_cost1 @ z
        p2_safety_cost = y @ safety_cost2 @ z

        p1_rank_cost = y @ rank_cost1 @ z
        p2_rank_cost = y @ rank_cost2 @ z

        theoretical_values.append(p1_safety_cost)


if __name__ == '__main__':
    model_filename = "saved_game.npz"
    results_filename = "results_test.npz"
    # results_filename = "results.npz"
    verbose = False

    # load saved game
    stage_count, rank_penalty_lst, safety_penalty_lst, init_state, states, control_inputs, \
        rank_cost1, rank_cost2, safety_cost1, safety_cost2, dynamics, \
        aggressive_policy1, aggressive_policy2, conservative_policy1, conservative_policy2, \
        moderate_policy1, moderate_policy2 = read_npz_build(model_filename)

    init_state_index = array_find(init_state, states)
    player_pairs = [(aggressive_policy1, aggressive_policy2), (conservative_policy1, conservative_policy2),
                    (moderate_policy1, moderate_policy2), (moderate_policy1, conservative_policy2),
                    (moderate_policy1, aggressive_policy2), (conservative_policy1, aggressive_policy2)]

    # must match directory name for visuals
    pair_labels = ["Aggressive-Aggressive", "Conservative-Conservative",
                   "Moderate-Moderate", "Moderate-Conservative",
                   "Moderate-Aggressive", "Conservative-Aggressive"]

    runs = 100000
    average_game_values = np.zeros((len(player_pairs), 2, 2))
    states_played = np.zeros((len(player_pairs), runs, stage_count+2), dtype=int)

    for i in range(len(player_pairs)):
        pair = player_pairs[i]
        label = pair_labels[i]
        policy1 = pair[0]
        policy2 = pair[1]
        total_values = np.zeros((2, 2))

        if verbose:
            print("new_pair")

        for j in range(runs):

            u, d, run_states_played = play_game(policy1, policy2, dynamics, stage_count, init_state_index)
            states_played[i, j] = run_states_played

            game_values = find_values(run_states_played, u, d, rank_cost1, rank_cost2, safety_cost1, safety_cost2)
            total_values += game_values

            if verbose:
                print(j)
                print("Control Inputs")
                print('u =', u)
                print('d =', d)
                print('\n')

                print("States Played")
                for k in range(len(run_states_played)):
                    print("Stage {} =".format(k))
                    print(states[run_states_played[k]])
                print('\n')
                print('Game values = ', game_values)

        average_cost = total_values / runs
        average_game_values[i] = average_cost

    saved_variables = (average_game_values, states_played, states, pair_labels)
    write_npz_play(results_filename, saved_variables)
    print("The End")
