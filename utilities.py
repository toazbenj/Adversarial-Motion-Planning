"""
Utilities

Basic functions for doing game theory calculations and more common drag race game
helper functions. Most assume game has separate cost matrices for each player, each player is
a minimizer.
"""
import csv
import ast
import numpy as np
from scipy.optimize import linprog

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

    # next state check
    next_state = dynamics(state, control_input1, control_input2)
    if next_state[0][0] == next_state[0][1] and next_state[1][0] == next_state[1][1]:
        return True

    # vehicles not turning
    if control_input1[0] == 0 and control_input2[0] == 0:
        return False

    # maneuver check
    is_same_lane_change = (control_input1[0]) == (-control_input2[0])
    is_same_speed = (control_input1[1] + state[2][0]) == (control_input2[1] + state[2][1])
    is_same_distance = state[0][0] == state[0][1]
    if is_same_lane_change and is_same_speed and is_same_distance:
        return True

    return False


def safety_cost(state, control_input1, control_input2, penalty_lst):
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

    # check for collisions
    if collision_check(state, control_input1, control_input2):
        return penalty_lst[-1], penalty_lst[-1]
    else:
        # calculate safety
        safety = [0, 0]
        for i in range(0, control_input.ndim):
            risk = 0
            # maintain speed
            if control_input[1][i] == 0:
                risk += penalty_lst[0]
            # decelerate
            elif control_input[1][i] < 0:
                risk += penalty_lst[1]
            # accelerate
            elif control_input[1][i] > 0:
                risk += penalty_lst[2]
            # turn
            if control_input[0][i] != 0:
                risk += penalty_lst[3]

            safety[i] = risk

        return safety[0], safety[1]


def rank_cost(state, control_input1, control_input2, penalty_lst):
    """
    Compute the cost matrix associated with the current game state for each player

    :param state: matrix of distance, lane, and velocity for each player
    :param control_input1: matrix of lane change and acceleration for player 1
    :param control_input2: matrix of lane change and acceleration for player 2
    :param penalty_lst: list of costs associated for maintaining speed, deceleration,
    acceleration,  turns, and collisions
    :return: single rank cost for both players, min and max with respect to player 1 position
    """

    # check for collisions
    if collision_check(state, control_input1, control_input2):
        rank_cost = np.array([penalty_lst[-1], penalty_lst[-1]])
    else:
        # calculate rank
        next_state = dynamics(state, control_input1, control_input2)
        p1_dist = next_state[0][0]
        p2_dist = next_state[0][1]
        rank_cost = np.array([p2_dist-p1_dist, p1_dist-p2_dist])

        p1_lane = next_state[1][0]
        p2_lane = next_state[1][1]

        if p1_lane == p2_lane:
            rank_cost *= 2

    return rank_cost[0], rank_cost[1]


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


def generate_costs(state_lst, control_input_lst, penalty_lst, cost_function):
    """
    Calculate stage cost given each state and control input

    :param state_lst: list of state arrays
    :param control_input_lst: list of all possible control input arrays of lane change and acceleration values
    :return: tensor of stage cost indexed by each state/control input
    """

    cost1 = np.empty((len(state_lst), len(control_input_lst),
                                       len(control_input_lst)), dtype=int) * np.nan
    cost2 = cost1.copy()

    # for k in range(stage_count):
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            for l in range(len(control_input_lst)):

                next_state_mat = dynamics(state_lst[i], control_input_lst[j], control_input_lst[l])
                next_state_index = array_find(next_state_mat, state_lst)
                if next_state_index != -1:
                    cost1[i, j, l], cost2[i, j, l] = cost_function(state_lst[i], control_input_lst[j],
                                                               control_input_lst[l], penalty_lst)

    return cost1, cost2


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


def expand_mat(small_mat, big_mat):
    """
    Helper function for making small matrix match dimensions of large one by repeating values
    :param small_mat: numpy array to expand
    :param big_mat: numpy array to match
    :return: expanded numpy array
    """
    shape_tup = big_mat.shape
    repeat_int = np.prod(shape_tup) // np.prod(small_mat.shape)
    expanded_mat = np.repeat(small_mat[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)
    return expanded_mat


def mixed_policy_2d(payoff_matrix, iterations=5000, is_min_max=True):
    """
    Calculate the mixed policies and values for each player
    :param payoff_matrix: game matrix with cost info
    :param is_min_max: if player 1 is minimizer or player 2 boolean
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
        if is_min_max:
            active_col = np.argmin(col_cum_payoff)
        else:
            active_col = np.argmax(col_cum_payoff)

        # Update column count and cumulative payoffs
        colcnt[active_col] += 1
        row_cum_payoff += transpose[active_col]

        # Choose the row with the maximum cumulative payoff
        if is_min_max:
            active_row = np.argmax(row_cum_payoff)
        else:
            active_row = np.argmin(row_cum_payoff)

    value_of_game = (np.max(row_cum_payoff) + np.min(col_cum_payoff)) / 2.0 / iterations
    return rowcnt/iterations, colcnt/iterations, round(value_of_game, 2)


def check_end_state(state_idx, state_lst, stage_count):
    state = state_lst[state_idx]
    if state[0][0] == stage_count + 1:
        return True
    elif state[0][1] == stage_count + 1:
        return True
    else:
        return False


def mixed_policy_3d(total_cost, state_lst, stage_count, is_min_max=True):
    """
    Find mixed saddle point game value for every state
    :param total_cost: 3D cost array of state x control input x control input
    :return: cost to go array of state x 1
    """

    num_states = total_cost.shape[0]
    ctg = np.zeros(num_states)
    row_policy = np.zeros((num_states, total_cost.shape[1]))
    col_policy = np.zeros((num_states, total_cost.shape[1]))

    for state in range(num_states):
        # if state is an ending state, no decisions made
        if check_end_state(state, state_lst, stage_count):
            row_policy[state] = 0
            col_policy[state] = 0
            ctg[state] = 0
        else:
            clean_mat = clean_matrix(total_cost[state])
            small_row_policy, small_col_policy, ctg[state] = mixed_policy_2d(clean_mat, is_min_max=is_min_max)

            row_policy[state] = remap_values(total_cost[state], small_row_policy)
            col_policy[state] = remap_values(total_cost[state], small_col_policy, is_row=False)

    return np.around(row_policy,2), np.around(col_policy,2), ctg


def bimatrix_mixed_policy(total_cost1, total_cost2, state_lst, stage_count):
    """
    Find mixed saddle point game value for every state
    :param total_cost: 3D cost array of state x control input x control input
    :return: cost to go array of state x 1
    """

    num_states = total_cost1.shape[0]
    ctg1 = np.zeros(num_states)
    ctg2 = np.zeros(num_states)
    row_policy = np.zeros((num_states, total_cost1.shape[1]))
    col_policy = np.zeros((num_states, total_cost2.shape[1]))

    for state in range(num_states):
        # if state is an ending state, no decisions made
        if check_end_state(state, state_lst, stage_count):
            row_policy[state] = 0
            col_policy[state] = 0
            ctg1[state] = 0
            ctg2[state] = 0
        else:
            cost1 = clean_matrix(total_cost1[state])
            cost2 = clean_matrix(total_cost2[state])
            small_row_policy, small_col_policy, ctg1[state], ctg2[state] = scipy_solve(cost1, cost2)

            row_policy[state] = remap_values(total_cost1[state], small_row_policy)
            col_policy[state] = remap_values(total_cost2[state], small_col_policy, is_row=False)

    return np.around(row_policy,2), np.around(col_policy,2), ctg1, ctg2


def scipy_solve(A, B):
    m, n = A.shape

    # Construct the objective function vector
    c = np.zeros(m + n + 2)
    c[-2:] = -1

    # Construct the inequality constraint matrix Ain and vector bin
    Ain_top = np.hstack((np.zeros((m, m)), -A, np.ones((m, 1)), np.zeros((m, 1))))
    Ain_bottom = np.hstack((-B.T, np.zeros((n, n + 1)), np.ones((n, 1))))
    Ain = np.vstack((Ain_top, Ain_bottom))
    bin = np.zeros(m + n)

    # Construct the equality constraint matrix Aeq and vector beq
    Aeq = np.zeros((2, m + n + 2))
    Aeq[0, :m] = 1
    Aeq[1, m:m + n] = 1
    beq = np.array([1, 1])

    # Define the bounds for the variables
    bounds = [(0, 1)] * (m + n) + [(-np.inf, np.inf), (-np.inf, np.inf)]

    # Solve the linear program
    result = linprog(c, A_ub=Ain, b_ub=bin, A_eq=Aeq, b_eq=beq, bounds=bounds)

    if result.success:
        x = result.x
        y = x[:m]
        z = x[m:m + n]
        p = x[m + n]
        q = x[m + n + 1]
        return y, z, p, q
    else:
        raise ValueError("Linear programming did not converge")


def clean_matrix(mat):
    """Remove rows/cols with all NaNs, keep matrix shape
    :param mat: numpy array
    :return: numpy array with no nans, retains relative position of real values
    """
    mat = mat[~np.isnan(mat).all(axis=1)]
    mat = mat[:, ~np.isnan(mat).all(axis=0)]
    return mat


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


def remap_values(mat, small_arr, is_row=True):
    """
    Map values from small arr to large arr, each large arr index corresponds to non nan value
    in either row or col of mat
    :param mat: mat of cost values
    :param small_arr: list of probabilities from cleaned payoff matrix
    :param is_row: policies for row or col
    :return: arr with non nan value indexes with probabilities, rest set to 0
    """
    large_arr = np.zeros(mat.shape[1])
    small_lst = list(small_arr)
    if is_row:
        mapping_arr = np.isfinite(mat[:, ~np.isnan(mat).all(axis=0)])[:,0]
    else:
        mapping_arr = np.isfinite(mat[~np.isnan(mat).all(axis=1)])[0]

    for i in range(len(large_arr)):
        if mapping_arr[i]:
            large_arr[i] = small_lst.pop(0)
        i += 1
    return large_arr


# def read_csv_to_variables(filename):
#     """
#     Load variables from csv file
#     :param filename: name of data file str
#     :return: tuple of offline calculated game variables
#     """
#     with open(filename, 'r') as csvfile:
#         csvreader = csv.reader(csvfile)
#
#         # Skip header
#         next(csvreader)
#
#         data_dict = {}
#         for row in csvreader:
#             key = row[0]
#             values = row[1:]
#             if len(values) == 1:
#                 # Convert single value to int or float
#                 value = values[0]
#                 if value.replace('.', '', 1).isdigit():
#                     data_dict[key] = float(value) if '.' in value else int(value)
#                 else:
#                     # Safely parse string representation of lists using ast.literal_eval
#                     data_dict[key] = ast.literal_eval(value)
#             else:
#                 # Handle lists of values
#                 data_dict[key] = [float(v) if '.' in v else np.array(v) for v in values]
#
#     # Assign variables from the dictionary
#     stage_count = data_dict['stage_count']
#     penalty_lst = data_dict['penalty_lst']
#
#     init_state = np.array(data_dict['init_state']).reshape((3, 2))
#     states = np.array(data_dict['states']).reshape((-1, 2))  # Adjust shape as per your actual states' shape
#     control_inputs = np.array(data_dict['control_inputs']).reshape(
#         (-1, 2))  # Adjust shape as per your actual control inputs' shape
#
#     costs1 = np.array(data_dict['costs1']).reshape((-1, 3))  # Adjust shape as per your actual costs1 shape
#     costs2 = np.array(data_dict['costs2']).reshape((-1, 3))  # Adjust shape as per your actual costs2 shape
#
#     dynamics = np.array(data_dict['dynamics']).reshape((-1, 2, 2))  # Adjust shape as per your actual dynamics shape
#
#     ctg1 = np.array(data_dict['ctg1']).reshape((-1, 2))  # Adjust shape as per your actual ctg1 shape
#     ctg2 = np.array(data_dict['ctg2']).reshape((-1, 2))  # Adjust shape as per your actual ctg2 shape
#     policy1 = np.array(data_dict['policy1']).reshape((-1, 2))  # Adjust shape as per your actual policy1 shape
#     policy2 = np.array(data_dict['policy2']).reshape((-1, 2))  # Adjust shape as per your actual policy2 shape
#
#     return (stage_count, penalty_lst, init_state, states, control_inputs, costs1, costs2, dynamics, ctg1, ctg2, policy1,
#             policy2)

def read_csv_to_variables(filename):
    """
    Load variables from csv file
    :param filename: name of data file str
    :return: tuple of offline calculated game variables
    """
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip header
        next(csvreader)

        data_dict = {}
        for row in csvreader:
            key = row[0]
            values = row[1:]
            if len(values) == 1:
                # Convert single value to int or float
                value = values[0]
                if value.replace('.', '', 1).isdigit():
                    data_dict[key] = float(value) if '.' in value else int(value)
                else:
                    # Safely parse string representation of lists using ast.literal_eval
                    data_dict[key] = ast.literal_eval(value)
            else:
                # Handle lists of values correctly
                parsed_values = []
                for v in values:
                    if v.startswith('['):
                        parsed_values.append(np.array(ast.literal_eval(v)))
                    elif '.' in v:
                        parsed_values.append(float(v))
                    else:
                        parsed_values.append(int(v))
                data_dict[key] = np.array(parsed_values)

    # Assign variables from the dictionary
    stage_count = data_dict['stage_count']
    penalty_lst = data_dict['penalty_lst']

    init_state = np.array(data_dict['init_state']).reshape((3, 2))
    states = np.concatenate([np.array(ast.literal_eval(v)) for v in data_dict['states']])
    control_inputs = np.array(data_dict['control_inputs']).reshape(
        (-1, 2))  # Adjust shape as per your actual control inputs' shape

    costs1 = np.array(data_dict['costs1']).reshape((-1, 3))  # Adjust shape as per your actual costs1 shape
    costs2 = np.array(data_dict['costs2']).reshape((-1, 3))  # Adjust shape as per your actual costs2 shape

    dynamics = np.array(data_dict['dynamics']).reshape((-1, 2, 2))  # Adjust shape as per your actual dynamics shape

    ctg1 = np.array(data_dict['ctg1']).reshape((-1, 2))  # Adjust shape as per your actual ctg1 shape
    ctg2 = np.array(data_dict['ctg2']).reshape((-1, 2))  # Adjust shape as per your actual ctg2 shape
    policy1 = np.array(data_dict['policy1']).reshape((-1, 2))  # Adjust shape as per your actual policy1 shape
    policy2 = np.array(data_dict['policy2']).reshape((-1, 2))  # Adjust shape as per your actual policy2 shape

    return (stage_count, penalty_lst, init_state, states, control_inputs, costs1, costs2, dynamics, ctg1, ctg2, policy1, policy2)


def write_variables_to_csv(filename, stage_count, penalty_lst, init_state, states, control_inputs, costs1, costs2,
                           dynamics, ctg1, ctg2, policy1, policy2):
    """
    Write all offline calculated data to csv file
    :param filename: name of data file
    :param stage_count: int
    :param penalty_lst: lst
    :param init_state: np array
    :param states: array of all possible states
    :param control_inputs: list of all control inputs
    :param costs1: state x ctrl1 x ctrl2 array of costs
    :param costs2: see above
    :param dynamics:state x crtl1 x crtl2 array of state indices
    :param ctg1: cost to go for player 1, stage x states
    :param ctg2: cost to go for player 2
    :param policy1: round x state x control inputs array of mixed policies for player 1
    :param policy2: see above
    """

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Variable', 'Value'])
        csvwriter.writerow(['stage_count', stage_count])
        csvwriter.writerow(['penalty_lst', penalty_lst])
        csvwriter.writerow(['init_state'] + init_state.flatten().tolist())
        csvwriter.writerow(['states'] + states)
        csvwriter.writerow(['control_inputs'] + control_inputs)
        csvwriter.writerow(['costs1'] + costs1.flatten().tolist())
        csvwriter.writerow(['costs2'] + costs2.flatten().tolist())
        csvwriter.writerow(['dynamics'] + dynamics.flatten().tolist())
        csvwriter.writerow(['ctg1'] + ctg1.flatten().tolist())
        csvwriter.writerow(['ctg2'] + ctg2.flatten().tolist())
        csvwriter.writerow(['policy1'] + policy1.flatten().tolist())
        csvwriter.writerow(['policy2'] + policy2.flatten().tolist())