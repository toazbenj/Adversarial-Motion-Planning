"""
Cost_to_go_experimental.py
Created by Ben Toaz on 6-7-24

Small scale implementation of matlab algorithm described in chapter 17 of Hespanha's NONCOOPERATIVE GAME THEORY.
Hard coded values for 2 rounds, 2 players
"""

import numpy as np

def collision_check(state, control_input1, control_input2):
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
    if control_input1[0] == 0 and control_input2[0] == 0:
        return False

    is_same_lane_change = (control_input1[0]) == (-control_input2[0])
    is_same_speed = (control_input1[1] + state[2][0]) == (control_input2[1] + state[2][1])
    is_same_distance = state[0][0] == state[0][1]
    if is_same_lane_change and is_same_speed and is_same_distance:
        return True

    return False


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
        rank_cost = 0.5
    else:
        # calculate rank
        p1_dist = state[0][0] + control_input1[1]
        p2_dist = state[0][1] + control_input2[1]
        if p1_dist > p2_dist:
            rank_cost = 0
        elif p1_dist < p2_dist:
            rank_cost = 1

    return rank_cost


def generate_states(k):
    """
    Calculate all possible states over k rounds
    :param k: the number of stages in game
    :return: list of all possible numpy array states
    """
    # generate all possible states for a player
    player_state_lst = []
    for distance in range(0, k+1):
        for lane in range(0, 2):
            for velocity in range(0, 2):
                state = [distance, lane, velocity]
                player_state_lst.append(state)

    # make list of all combinations for both players
    state_lst = []
    for player_state1 in player_state_lst:
        for player_state2 in player_state_lst:
            state = np.array([player_state1, player_state2])
            state_lst.append(state.T)

    return state_lst


def generate_control_inputs():
    """
    Make list of all possible player actions given space/movement constraints
    :return: list of control input numpy arrays
    """
    control_input_lst = []
    lane_maneuver_range = range(-1, 2)
    acceleration_maneuver_range = range(-1, 2)
    for i in lane_maneuver_range:
        for j in acceleration_maneuver_range:
            control_input_lst.append(np.array([i, j]))

    return control_input_lst


def generate_costs(state_lst, control_input_lst):
    """
    Calculate stage cost given each state and control input
    :return: tensor of stage cost indexed by each state/control input
    """

    cost_lookup_mat = np.empty((len(state_lst), len(control_input_lst),
                                       len(control_input_lst)), dtype=int) * np.nan
    # for k in range(stage_count):
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            for l in range(len(control_input_lst)):

                cost_lookup_mat[i, j, l] = rank_cost(state_lst[i], control_input_lst[j], control_input_lst[l])

    return cost_lookup_mat


def dynamics(state, control_input1, control_input2):
    """
    Calculate next state from given current state and control input
    :param state: matrix of distance, lane, and velocity for each player
    :param control_input1: player one lane and velocity change matrix
    :param control_input2: player two lane and velocity change matrix
    :return: next state matrix
    """
    control_input_trans = np.array([control_input1.T, control_input2.T])
    control_input = control_input_trans.T

    A = np.array([[1, 0, 1],
                  [0, 1, 0],
                  [0, 0, 1]])
    B = np.array([[0, 1],
                  [1, 0],
                  [0, 1]])
    next_state = np.dot(A, state) + np.dot(B, control_input)

    return next_state


def array_find(value_array, search_lst):
    for idx, item in enumerate(search_lst):
        if np.array_equal(value_array, item):
            return idx
    return -1


def generate_dynamics(state_lst, control_input_lst):
    """
    Make lookup table for next state given current state and player actions, implemented as nested dictionary
    :param state_lst:
    :param control_input_lst:
    :return:
    """
    dynamics_lookup_mat = np.empty((len(state_lst), len(control_input_lst),
                                    len(control_input_lst)), dtype=int) * np.nan
    # for k in range(stage_count):
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            for l in range(len(control_input_lst)):
                next_state_mat = dynamics(state_lst[i], control_input_lst[j], control_input_lst[l])
                next_state_index = array_find(next_state_mat, state_lst)
                if next_state_index != -1:
                    dynamics_lookup_mat[i, j, l] = next_state_index

    return dynamics_lookup_mat


def generate_cost_to_go(k, cost):
    # Initialize V with zeros
    V = np.zeros((k + 2, len(cost)))

    # Iterate backwards from k to 1
    for stage in range(k, -1, -1):
        # Calculate Vminmax and Vmaxmin
        V_last = V[stage + 1]
        shape_tup = cost.shape
        repeat_int = np.prod(shape_tup) // np.prod(V_last.shape)
        V_expanded = np.repeat(V_last[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        Vminmax = np.min(np.max(cost[stage] + V_expanded, axis=1), axis=1)
        Vmaxmin = np.max(np.min(cost[stage] + V_expanded, axis=2), axis=1)

        # Check if saddle-point can be found
        if not np.array_equal(Vminmax, Vmaxmin):
            print("Must find mixed policy")

        # Assign Vminmax to V[k-1]
        V[stage] = Vminmax

    return V


def optimal_actions(k, cost, ctg, dynamics, initial_state):
    control1 = np.zeros(k+1, dtype=int)
    control2 = np.zeros(k+1, dtype=int)
    states_played = np.zeros(k+2, dtype=int)
    states_played[0] = initial_state

    for stage in range(k+1):
        V_last = ctg[stage+1]
        shape_tup = cost.shape
        repeat_int = np.prod(shape_tup)//np.prod(V_last.shape)
        V_expanded = np.repeat(V_last[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        control1[stage] = np.argmin(np.max(cost[states_played[stage]] + V_expanded[states_played[stage]], axis=1), axis=0)
        control2[stage] = np.argmax(np.min(cost[states_played[stage]] + V_expanded[states_played[stage]], axis=0), axis=0)

        # likely incorrect dynamics or cost, [0,2,2] yields NaN
        # cost function does not take control input into account
        states_played[stage + 1] = dynamics[states_played[stage], control1[stage], control2[stage]]

    return control1, control2, states


if __name__ == '__main__':
    stage_count = 1

    states = generate_states(stage_count)
    control_inputs = generate_control_inputs()

    costs = generate_costs(states, control_inputs)
    dynamics = generate_dynamics(states, control_inputs)
    # print(dynamics)

    ctg = generate_cost_to_go(stage_count, costs)
    for k in range(stage_count + 2):
        print("V[{}] = {}".format(k, ctg[k]))

    init = 2
    u, d, states_played = optimal_actions(stage_count, costs, ctg, dynamics, init)
    print('u =', u)
    print('d =', d)
    print('x =', states_played)