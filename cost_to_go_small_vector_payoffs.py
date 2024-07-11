"""
Cost to Go Small Vector Payoffs
Created by Ben Toaz on 6-7-24

2 player drag race with different payoffs for each player, security policies for safety and rank bimatrix games with
both players as minimizers
"""

import numpy as np
from graphics import plot_race
from utilities import *
import random

def generate_cost_to_go(stage_count, costs1, costs2):
    """
    Calculates cost to go for each state at each stage
    :param k: stage count int
    :param cost: cost array of state x control input x control input
    :return: cost to go of stage x state
    """
    # Initialize V with zeros
    V1 = np.zeros((stage_count + 2, len(costs1)))
    V2 = np.zeros((stage_count + 2, len(costs2)))

    # Iterate backwards from k to 1
    for stage in range(stage_count, -1, -1):
        V_expanded1 = expand_mat(V1[stage + 1], costs1)
        V_expanded2 = expand_mat(V2[stage + 1], costs2)

        Vminmax1 = np.nanmin(np.nanmax(costs1 + V_expanded1, axis=1), axis=1)
        Vminmax2 = np.nanmin(np.nanmax(costs2 + V_expanded2, axis=2), axis=1)

        V1[stage] = Vminmax1
        V2[stage] = Vminmax2

    return V1, V2


def generate_cost_to_go_mixed(stage_count, costs1, costs2, control_inputs, state_lst):
    """
    Calculates cost to go for each state at each stage
    :param k: stage count int
    :param cost: cost array of state x control input x control input
    :return: cost to go of stage x state
    """
    # Initialize cost to go with zeros
    V1 = np.zeros((stage_count + 2, len(costs1)))
    V2 = np.zeros((stage_count + 2, len(costs1)))
    policy1 = np.zeros((stage_count + 1, len(state_lst), len(control_inputs)))
    policy2 = np.zeros((stage_count + 1, len(state_lst), len(control_inputs)))

    # Iterate backwards from k to 1
    for stage in range(stage_count, -1, -1):
        combined_cost1 = expand_mat(V1[stage + 1], costs1) + costs1
        combined_cost2 = expand_mat(V2[stage + 1], costs2) + costs2

        # policy1[stage], _, mixed_value1 = mixed_policy_3d(combined_cost1, state_lst, stage_count)
        # _, policy2[stage], mixed_value2 = mixed_policy_3d(combined_cost2, state_lst, stage_count)

        policy1[stage], policy2[stage], mixed_value1, mixed_value2 = bimatrix_mixed_policy(combined_cost1,
                                                                                           combined_cost2,
                                                                                           state_lst, stage_count)

        V1[stage] = mixed_value1
        V2[stage] = mixed_value2

    return V1, V2, policy1, policy2


def optimal_actions(stage_count, costs1, costs2, ctg1, ctg2, dynamics, init_state_index):
    """
    Given initial state, play actual game, calculate best control inputs and tabulate state at each stage
    :param stage_count: stage count
    :param cost: cost array of state x control input 1 x control input 2
    :param ctg: cost to go array of stage x state
    :param dynamics: next state array given control inputs of state x control input 1 x control input 2
    :param initial_state: index of current state int
    :return: list of best control input indicies for each player, states played in the game
    """
    control1 = np.zeros(stage_count+1, dtype=int)
    control2 = np.zeros(stage_count+1, dtype=int)
    states_played = np.zeros(stage_count + 2, dtype=int)
    states_played[0] = init_state_index

    for stage in range(stage_count + 1):
        V_expanded1 = expand_mat(ctg1[stage + 1], costs1)
        V_expanded2 = expand_mat(ctg2[stage + 1], costs2)

        control1[stage] = np.nanargmin(
            np.nanmax(costs1[states_played[stage]] + V_expanded1[states_played[stage]], axis=1), axis=0)
        control2[stage] = np.nanargmin(
            np.nanmax(costs2[states_played[stage]] + V_expanded2[states_played[stage]], axis=0), axis=0)

        states_played[stage + 1] = dynamics[states_played[stage], control1[stage], control2[stage]]

    return control1, control2, states_played


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
    values = np.array([[0, 0], [0, 0]])
    for idx in range(len(states_played)-1):
        round_rank_cost1 = rank_cost1[states_played[idx], u[idx], d[idx]]
        round_rank_cost2 = rank_cost2[states_played[idx], u[idx], d[idx]]

        round_safety_cost1 = safety_cost1[states_played[idx], u[idx], d[idx]]
        round_safety_cost2 = safety_cost2[states_played[idx], u[idx], d[idx]]

        values[0] += np.array([int(round_rank_cost1), int(round_safety_cost1)])
        values[1] += np.array([int(round_rank_cost2), int(round_safety_cost2)])
    return values


if __name__ == '__main__':
    is_new_game = False
    filename = "current_game.npz"

    if is_new_game:
        stage_count = 0
        # maintain speed, decelerate, accelerate, turn, tie, collide
        rank_penalty_lst = [0, 1, 2, 1, 5, 10]
        safety_penalty_lst = [0, 3, 3, 3, 5, 20]

        init_state = np.array([[0, 0],
                               [0, 1],
                               [0, 0]])

        states = generate_states(stage_count)
        control_inputs = generate_control_inputs()

        rank_cost1, rank_cost2 = generate_costs(states, control_inputs, rank_penalty_lst, rank_cost)
        safety_cost1, safety_cost2 = generate_costs(states, control_inputs, safety_penalty_lst, safety_cost)

        dynamics = generate_dynamics(states, control_inputs)

        rank_ctg1, rank_ctg2, aggressive_policy1, aggressive_policy2 =\
            generate_cost_to_go_mixed(stage_count,
                                      rank_cost1, rank_cost2,
                                      control_inputs, states)
        safety_ctg1, safety_ctg2, conservative_policy1, conservative_policy2 =\
            generate_cost_to_go_mixed(stage_count,
                                      safety_cost1, safety_cost2,
                                      control_inputs, states)

        moderate_policy1, moderate_policy2 = generate_moderate_policies(aggressive_policy1, aggressive_policy2,
                                                                        conservative_policy1, conservative_policy2)

        write_variables_to_npz(filename, (stage_count, rank_penalty_lst, safety_penalty_lst, init_state, states,
                                          control_inputs, rank_cost1, rank_cost2, safety_cost1, safety_cost2, dynamics,
                                          aggressive_policy1, aggressive_policy2,
                                          conservative_policy1, conservative_policy2,
                                          moderate_policy1, moderate_policy2))
    else:
        # load saved game
        stage_count, rank_penalty_lst, safety_penalty_lst, init_state, states, control_inputs, \
         rank_cost1, rank_cost2, safety_cost1, safety_cost2, dynamics, \
         aggressive_policy1, aggressive_policy2, conservative_policy1, conservative_policy2, \
         moderate_policy1, moderate_policy2 = read_npz_to_variables(filename)

    # print("Cost to Go")
    # for k in range(stage_count + 2):
    #     print("V1[{}] = {}".format(k, ctg1[k]))
    # print('\n')
    # for k in range(stage_count + 2):
    #     print("V2[{}] = {}".format(k, ctg2[k]))
    # print('\n')
    #
    # print("Policies")
    # for k in range(stage_count + 1):
    #     print("y[{}] = {}".format(k, policy1[k]))
    # print('\n')
    # for k in range(stage_count + 1):
    #     print("z[{}] = {}".format(k, policy2[k]))
    # print('\n')

    init_state_index = array_find(init_state, states)
    player_pairs = [(aggressive_policy1, aggressive_policy2), (conservative_policy1, conservative_policy2),
                    (moderate_policy1, moderate_policy2), (moderate_policy1, conservative_policy1),
                    (moderate_policy1, aggressive_policy2), (conservative_policy1, aggressive_policy2)]
    pair_labels = ["Aggressive vs. Aggressive", "Conservative vs. Conservative",
                   "Moderate vs. Moderate", "Moderate vs. Conservative",
                   "Moderate vs. Aggressive", "Conservative vs Aggressive"]
    runs = 1
    for pair, label in zip(player_pairs, pair_labels):
        policy1 = pair[0]
        policy2 = pair[1]
        for i in range(runs):
            u, d, states_played = play_game(policy1, policy2, dynamics, stage_count, init_state_index)

            print("Control Inputs")
            print('u =', u)
            print('d =', d)
            print('\n')

            print("States Played")
            for i in range(len(states_played)):
                print("Stage {} =".format(i))
                print(states[states_played[i]])
            print('\n')

            game_values = find_values(states_played, u, d, rank_cost1, rank_cost2, safety_cost1, safety_cost2)
            print('Game values = ', game_values)

            plot_race(states_played, states, label, i)
print("The End")