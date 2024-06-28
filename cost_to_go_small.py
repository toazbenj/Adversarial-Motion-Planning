"""
Cost to Go Small
Created by Ben Toaz on 6-7-24

Small scale implementation of matlab algorithm described in chapter 17 of Hespanha's NONCOOPERATIVE GAME THEORY.
Scenario is only for 2 players
"""

import numpy as np
from graphics import plot_race
from utilities import (collision_check, array_find, dynamics, generate_dynamics, mixed_policy_3d, generate_states,
                       generate_control_inputs)
def safety_cost(state, control_input1, control_input2):
    """
    Compute the cost matrix associated with the current game state for each player

    :param state: matrix of distance, lane, and velocity for each player
    :param control_input: matrix of lane change and acceleration for each player
    :param penalty_lst: list of costs associated for maintaining speed, deceleration,
    acceleration,  turns, and collisions
    :return: cost matrix with for safety and rank objectives for each player
    """

    control_input_trans = np.array([control_input1.T, control_input2.T])
    control_input = control_input_trans.T
    penalty_lst = [0, 1, 2, 1, 10]

    # check for collisions
    # enforce boundary conditions for the track also
    collision_penalty_int = penalty_lst[-1]
    if collision_check(state, control_input1, control_input2):
        return np.array([[collision_penalty_int, collision_penalty_int],
                        [collision_penalty_int, collision_penalty_int]])
    else:
        # calculate rank
        ranks = [1, 1]
        p1_dist = state[0][0] + control_input1[0][1]
        p2_dist = state[0][1] + control_input2[0][1]
        if p1_dist > p2_dist:
            ranks = [0, 1]
        elif p1_dist < p2_dist:
            ranks = [1, 0]

        # calculate safety
        safety = [0, 0]
        for i in range(0, control_input.ndim-1):
            risk = 0
            # maintain speed
            if control_input[0][1][i] == 0:
                risk += penalty_lst[0]
            # decelerate
            elif control_input[0][1][i] < 0:
                risk += penalty_lst[1]
            # accelerate
            elif control_input[0][1][i] > 0:
                risk += penalty_lst[2]
            # turn
            if control_input[0][0][i] != 0:
                risk += penalty_lst[3]

            safety[i] = risk

        return np.array([safety,
                        ranks])


def rank_cost(state, control_input1, control_input2):
    """
    Compute the cost matrix associated with the current game state for each player

    :param state: matrix of distance, lane, and velocity for each player
    :param control_input: matrix of lane change and acceleration for each player
    :param penalty_lst: list of costs associated for maintaining speed, deceleration,
    acceleration,  turns, and collisions
    :return: single rank cost for both players, min and max with respect to player 1 position
    """

    penalty_lst = [0, 1, 2, 1, 10]

    # check for collisions
    # enforce boundary conditions for the track also
    collision_penalty_int = penalty_lst[-1]
    rank_cost = 0.5
    if collision_check(state, control_input1, control_input2):
        rank_cost = 10
    else:
        # calculate rank
        p1_dist = state[0][0] + control_input1[1]
        p2_dist = state[0][1] + control_input2[1]
        if p1_dist > p2_dist:
            rank_cost = 0
        elif p1_dist < p2_dist:
            rank_cost = 1

    return rank_cost


def generate_costs(state_lst, control_input_lst):
    """
    Calculate stage cost given each state and control input

    :param state_lst: list of state arrays
    :param control_input_lst: list of all possible control input arrays of lane change and acceleration values
    :return: tensor of stage cost indexed by each state/control input
    """

    cost_lookup_mat = np.empty((len(state_lst), len(control_input_lst),
                                       len(control_input_lst)), dtype=int) * np.nan
    # for k in range(stage_count):
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            for l in range(len(control_input_lst)):

                next_state_mat = dynamics(state_lst[i], control_input_lst[j], control_input_lst[l])
                next_state_index = array_find(next_state_mat, state_lst)
                if next_state_index != -1:
                    cost_lookup_mat[i, j, l] = rank_cost(state_lst[i], control_input_lst[j], control_input_lst[l])

    return cost_lookup_mat


def generate_cost_to_go(k, cost):
    """
    Calculates cost to go for each state at each stage
    :param k: stage count int
    :param cost: cost array of state x control input x control input
    :return: cost to go of stage x state
    """
    # Initialize V with zeros
    V = np.zeros((k + 2, len(cost)))

    # Iterate backwards from k to 1
    for stage in range(k, -1, -1):
        # Calculate Vminmax and Vmaxmin
        V_last = V[stage + 1]
        shape_tup = cost.shape
        repeat_int = np.prod(shape_tup) // np.prod(V_last.shape)
        V_expanded = np.repeat(V_last[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        Vminmax = np.min(np.max(cost + V_expanded, axis=1), axis=1)
        Vmaxmin = np.max(np.min(cost + V_expanded, axis=2), axis=1)

        # Check if saddle-point can be found
        if np.array_equal(Vminmax, Vmaxmin):
            # Assign Vminmax to V[k-1]
            V[stage] = Vminmax
        else:
            # print("Must find mixed policy")
            _, _, V[stage] = mixed_policy_3d(cost + V_expanded)

    return V


def optimal_actions(k, cost, ctg, dynamics, initial_state):
    """
    Given initial state, play actual game, calculate best control inputs and tabulate state at each stage
    :param k: stage count
    :param cost: cost array of state x control input 1 x control input 2
    :param ctg: cost to go array of stage x state
    :param dynamics: next state array given control inputs of state x control input 1 x control input 2
    :param initial_state: index of current state int
    :return: list of best control input indicies for each player, states played in the game
    """
    control1 = np.zeros(k+1, dtype=int)
    control2 = np.zeros(k+1, dtype=int)
    states_played = np.zeros(k+2, dtype=int)
    states_played[0] = initial_state

    for stage in range(k+1):
        V_last = ctg[stage+1]
        shape_tup = cost.shape
        repeat_int = np.prod(shape_tup)//np.prod(V_last.shape)
        V_expanded = np.repeat(V_last[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        control1[stage] = np.nanargmin(np.nanmax(cost[states_played[stage]] + V_expanded[states_played[stage]], axis=1),
                                    axis=0)
        control2[stage] = np.nanargmax(np.nanmin(cost[states_played[stage]] + V_expanded[states_played[stage]], axis=0),
                                    axis=0)

        states_played[stage + 1] = dynamics[states_played[stage], control1[stage], control2[stage]]

    return control1, control2, states_played


if __name__ == '__main__':
    stage_count = 0

    states = generate_states(stage_count)
    control_inputs = generate_control_inputs()

    costs = generate_costs(states, control_inputs)
    dynamics = generate_dynamics(states, control_inputs)
    # print(dynamics)

    ctg = generate_cost_to_go(stage_count, costs)
    for k in range(stage_count + 2):
        print("V[{}] = {}".format(k, ctg[k]))

    init_state = np.array([[0, 0],
                           [0, 1],
                           [0, 0]])
    init_state_index = array_find(init_state, states)
    u, d, states_played = optimal_actions(stage_count, costs, ctg, dynamics, init_state_index)
    print('u =', u)
    print('d =', d)
    print("States Played")
    for i in range(len(states_played)):
        print(i, states[states_played[i]])

    plot_race(states_played, states)