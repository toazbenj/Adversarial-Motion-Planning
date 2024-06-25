"""
Cost to Go Small Vector Payoffs
Created by Ben Toaz on 6-7-24

2 player drag race with different payoffs for each player
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
    rank_cost = np.array([[0.5, 0.5]])
    if collision_check(state, control_input1, control_input2):
        rank_cost = np.array([[10, 10]])
    else:
        # calculate rank
        p1_dist = state[0][0] + control_input1[1]
        p2_dist = state[0][1] + control_input2[1]
        if p1_dist > p2_dist:
            rank_cost = np.array([[0, 1]])
        elif p1_dist < p2_dist:
            rank_cost = np.array([[1, 0]])

    return rank_cost


def check_state(state):
    """
    Verifies if state is possible
    :param state: numpy array with state info
    :return: bool if state is possible
    """
    # invalid starting line conditions
    # position (wrong lane)
    if state[0][0] == 0 and state[1][0] == 1:
        return False
    elif state[0][1] == 0 and state[1][1] == 0:
        return False
    # velocity (initial speed)
    if state[0][0] == 0 and state[2][0] == 1:
        return False
    elif state[0][1] == 0 and state[2][1] == 1:
        return False
    else:
        return True


def generate_states(k):
    """
    Calculate all possible states over k rounds
    :param k: the number of stages in game
    :return: list of all possible numpy array states
    """
    # generate all possible states for a player
    player_state_lst = []
    for distance in range(0, k+2):
        for lane in range(0, 2):
            for velocity in range(0, 2):
                state = [distance, lane, velocity]
                player_state_lst.append(state)

    # make list of all combinations for both players
    state_lst = []
    cnt = 0
    for player_state1 in player_state_lst:
        for player_state2 in player_state_lst:
            state = np.array([player_state1, player_state2]).T

            if check_state(state):
                state_lst.append(state)
            # else:
            #     cnt +=1
            #     print(state,cnt,"\n")
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

    :param state_lst: list of state arrays
    :param control_input_lst: list of all possible control input arrays of lane change and acceleration values
    :return: tensor of stage cost indexed by each state/control input
    """

    cost_lookup_mat = np.empty((len(state_lst), len(control_input_lst),
                                       len(control_input_lst), 2), dtype=int) * np.nan
    # for k in range(stage_count):
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            for l in range(len(control_input_lst)):

                next_state_mat = dynamics(state_lst[i], control_input_lst[j], control_input_lst[l])
                next_state_index = array_find(next_state_mat, state_lst)
                if next_state_index != -1:
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
    """
    Find the index of the array within a list of arrays
    :param value_array: array to search for
    :param search_lst: list to search within
    :return: index of array, -1 if not found
    """
    for idx, item in enumerate(search_lst):
        if np.array_equal(value_array, item):
            return idx
    return -1


def generate_dynamics(state_lst, control_input_lst):
    """
    Make lookup table for next state given current state and player actions, implemented as nested dictionary
    :param state_lst: list of all possible state arrays
    :param control_input_lst: list of all possible control input arrays
    :return: array of dynamics, 3D, dimensions state, control input 1, control input 2
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


def mixed_policy_2d(payoff_matrix, iterations=5000):
    """
    Calculate the mixed policies and values for each player
    :param payoff_matrix: game matrix with cost info
    :param iterations: number of loops in calculation
    :return: player one and two policies as lists of floats, game value
    """
    payoff_matrix = np.array(payoff_matrix)
    transpose = payoff_matrix.T
    numrows, numcols = payoff_matrix.shape
    row_cum_payoff = np.zeros(numrows)
    col_cum_payoff = np.zeros(numcols)
    colcnt = np.zeros(numcols)
    rowcnt = np.zeros(numrows)
    active_row = 0

    # iterates through row/col combinations, selects best play for each player with different combinations of rows/col,
    # sums up number of times each row/col was selected and averages to find mixed policies
    for _ in range(iterations):
        # Update row count and cumulative payoffs
        rowcnt[active_row] += 1
        col_cum_payoff += payoff_matrix[active_row]

        # Choose the column with the minimum cumulative payoff
        active_col = np.argmin(col_cum_payoff)

        # Update column count and cumulative payoffs
        colcnt[active_col] += 1
        row_cum_payoff += transpose[active_col]

        # Choose the row with the maximum cumulative payoff
        active_row = np.argmax(row_cum_payoff)

    value_of_game = (np.max(row_cum_payoff) + np.min(col_cum_payoff)) / 2.0 / iterations
    return rowcnt / iterations, colcnt / iterations, round(value_of_game, 2)


def clean_matrix(mat):
    """Remove rows/cols with all NaNs
    :param mat: numpy array
    :return: numpy array with no nans, retains relative position of real values
    """
    mat = mat[~np.isnan(mat).all(axis=1)]
    mat = mat[:, ~np.isnan(mat).all(axis=0)]
    return mat


def mixed_policy_3d(total_cost):
    """
    Find mixed saddle point game value for every state
    :param total_cost: 3D cost array of state x control input x control input
    :return: cost to go array of state x 1
    """

    num_states = total_cost.shape[0]
    ctg = np.zeros(num_states)

    for state in range(num_states):
        clean_mat = clean_matrix(total_cost[state])
        _, _, ctg[state] = mixed_policy_2d(clean_mat)

    return ctg


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
            V[stage] = mixed_policy_3d(cost + V_expanded)

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


def play_game(u, d, init_state_index):
    pass


if __name__ == '__main__':
    stage_count = 1

    states = generate_states(stage_count)
    control_inputs = generate_control_inputs()

    costs1 = generate_costs(states, control_inputs, 1)
    costs2 = generate_costs(states, control_inputs, 2)
    dynamics = generate_dynamics(states, control_inputs)
    # print(dynamics)

    ctg1 = generate_cost_to_go(stage_count, costs1)
    ctg2 = generate_cost_to_go(stage_count, costs2)
    for k in range(stage_count + 2):
        print("V1[{}] = {}".format(k, ctg1[k]))
    for k in range(stage_count + 2):
        print("V1[{}] = {}".format(k, ctg2[k]))

    init_state = np.array([[0, 0],
                           [0, 1],
                           [0, 0]])
    init_state_index = array_find(init_state, states)
    u = optimal_actions(stage_count, costs1, ctg1, dynamics, init_state_index)
    d = optimal_actions(stage_count, costs2, ctg2, dynamics, init_state_index)
    states_played = play_game(u, d, init_state_index)
    print('u =', u)
    print('d =', d)
    print("States Played")
    for i in range(len(states_played)):
        print(i, states[states_played[i]])