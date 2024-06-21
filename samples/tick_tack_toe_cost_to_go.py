"""
TickTackToeCostToGo.py

Even more watered down ctg for drag race modeled after tic tac toe example in game theory text. Converted from Matlab.
Doesn't work
"""

import numpy as np
from cost_to_go_small import generate_dynamics, generate_control_inputs, generate_states, generate_costs, \
    dynamics

def check_state_valid(state, ctrl, k):
    for pos in state[0]:
        if pos > k or pos < 0:
            return False
    for lane in state[1]:
        if lane > state.ndim-1 or lane < 0:
            return False

    for velocity_count in range(state.ndim):
        if state[2][velocity_count] > 1 or state[2][velocity_count] < 0:
            return False
        # case of lane changes with no velocity
        if state[2][velocity_count] == 0 and ctrl[1][velocity_count] == 0 and ctrl[0][velocity_count] != 0:
            return False
    return True

def collision_check(state, control_input):
    """
    Compares player positions and returns whether they are in the same position or collided during the next maneuver
    :param state: matrix of distance, lane, and velocity for each player
    :param control_input: matrix of lane change and acceleration for each player
    :return: boolean
    """

    # pos check
    if state[0][0] == state[0][1] and state[1][0] == state[1][1]:
        return True

    # maneuver check
    # vehicles not turning
    if control_input[0][0] == 0 and control_input[0][1] == 0:
        return False

    # is_same_lane_change = (control_input[0][0]) == (-control_input[0][1])
    # is_same_speed = (control_input[1][0] + state[2][0]) == (control_input[1][1] + state[2][1])
    # is_same_distance = state[0][0] == state[0][1]
    # if is_same_lane_change and is_same_speed and is_same_distance:
    #     return True

    return False


def possible_states(stage_count, initial_state, control_inputs):
    """
    Possible states for each round
    :param states:
    :param k:
    :return:
    """
    possible_states = [[initial_state]]
    for k in range(1, stage_count+1):
        round_states = []
        for state in possible_states[k-1]:
            for ctrl in control_inputs:
                # check if last state was a collision, don't propagate collisions
                # not checking for collision in transit yet
                if collision_check(state, ctrl):
                    new_state = state
                    round_states.append(new_state)
                    break
                new_state = dynamics(state, ctrl)
                if check_state_valid(new_state, ctrl, k):
                    round_states.append(new_state)

        possible_states.append(round_states)
    return possible_states



# def generate_cost_to_go_ttt(k, cost, dynamics, states, control_inputs):
    # nX = len(states)  # Number of states
    # nU = lens(control_inputs)  # Number of actions (slots on the board)
    # V = [np.zeros_like(states[k])] * (K + 1)  # Initialize final stage cost-to-go values
    #
    # for k in range(K, 0, -1):
    #     newS_X, won_X, invalid_X = ttt_addX(S[k])  # Compute all next states for X
    #     newS_O, won_O, invalid_O = ttt_addO(S[k])  # Compute all next states for O
    #
    #     newV = np.zeros((nX, nU, nU), dtype=np.int8)  # Initialize new cost-to-go values
    #     for i in range(nX):
    #         for u_X in range(nU):
    #             for u_O in range(nU):
    #                 if invalid_X[i, u_X] or invalid_O[i, u_O]:
    #                     newV[i, u_X, u_O] = np.nan  # Invalid moves result in NaN
    #                 elif won_X[i, u_X]:
    #                     newV[i, u_X, u_O] = -1  # X wins
    #                 elif won_O[i, u_O]:
    #                     newV[i, u_X, u_O] = 1  # O wins
    #                 else:
    #                     exists_X, ndx_X = np.isin(newS_X[:, u_X], S[k + 1], assume_unique=True)
    #                     exists_O, ndx_O = np.isin(newS_O[:, u_O], S[k + 1], assume_unique=True)
    #                     if np.logical_and(exists_X[i], exists_O[i]):
    #                         newV[i, u_X, u_O] = V[k + 1][ndx_X[i]] + V[k + 1][ndx_O[i]]
    #                     else:
    #                         newV[i, u_X, u_O] = 0  # Default value if state doesn't exist
    #
    #     # Convert NaN to penalties for invalid moves
    #     newV[np.isnan(newV)] = -np.inf
    #
    #     # Compute the value for the current stage considering simultaneous play
    #     V[k] = np.max(np.min(newV, axis=2), axis=1)
    #
    # return V

if __name__ == '__main__':
    k =2
    states = generate_states(k)
    print("states ",states, "\n")

    control_inputs = generate_control_inputs()
    print("crtl ", control_inputs, "\n")

    init = np.array([[0,0],[0,1],[0,0]])
    possible_states = possible_states(k, init, control_inputs)
    print("possible states ", possible_states, "\n")

    costs = generate_costs(possible_states, control_inputs, k)
    print(costs, "\n")

    # ctg = generate_cost_to_go_ttt(2, costs, dynamics, states)
    # print(ctg, "\n")