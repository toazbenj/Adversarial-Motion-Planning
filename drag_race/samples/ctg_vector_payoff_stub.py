"""
Cost to Go Small Vector Payoffs
Created by Ben Toaz on 6-7-24

Testing with tug of war example, different payoffs for each player
"""

import numpy as np

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


def generate_cost_to_go(k, cost1, cost2):
    """
    Calculates cost to go for each state at each stage
    :param k: stage count int
    :param cost: cost array of state x control input x control input
    :return: cost to go of stage x state
    """
    # Initialize V with zeros
    V1 = np.zeros((k + 2, len(cost1[0])))
    V2 = np.zeros((k + 2, len(cost2[0])))

    # Iterate backwards from k to 1
    for stage in range(k, -1, -1):
        # Calculate Vminmax and Vmaxmin
        V_last1 = V1[stage + 1]
        shape_tup = cost1[0].shape
        repeat_int = np.prod(shape_tup) // np.prod(V_last1.shape)
        V_expanded1 = np.repeat(V_last1[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        V_last2 = V2[stage + 1]
        shape_tup = cost2[0].shape
        repeat_int = np.prod(shape_tup) // np.prod(V_last2.shape)
        V_expanded2 = np.repeat(V_last2[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        Vminmax1 = np.min(np.max(cost1[stage] + V_expanded1, axis=1), axis=1)
        Vminmax2 = np.min(np.max(cost2[stage] + V_expanded2, axis=2), axis=1)

        Vmaxmin1 = np.max(np.min(cost1[stage] + V_expanded1, axis=2), axis=1)
        Vmaxmin2 = np.max(np.min(cost2[stage] + V_expanded2, axis=1), axis=1)

        V1[stage] = Vminmax1
        V2[stage] = Vminmax2

        # Check if saddle-point can be found
        if np.array_equal(Vminmax1, Vmaxmin1, equal_nan=True):
            V1[stage] = Vminmax1
        else:
            V1[stage] = mixed_policy_3d(cost1 + V_expanded1)

        # Check if saddle-point can be found
        if np.array_equal(Vminmax2, Vmaxmin2, equal_nan=True):
            V2[stage] = Vminmax2
        else:
            V2[stage] = mixed_policy_3d(cost2 + V_expanded2)

    return V1, V2


def optimal_actions(k, cost1, ctg1, cost2, ctg2, dynamics, initial_state):
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
        V_last1 = ctg1[stage+1]
        shape_tup = cost1[stage].shape
        repeat_int = np.prod(shape_tup)//np.prod(V_last1.shape)
        V_expanded1 = np.repeat(V_last1[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        V_last2 = ctg2[stage + 1]
        shape_tup = cost2[stage].shape
        repeat_int = np.prod(shape_tup) // np.prod(V_last2.shape)
        V_expanded2 = np.repeat(V_last2[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        control1[stage] = np.nanargmin(np.nanmax(cost1[stage][states_played[stage]] + V_expanded1[states_played[stage]], axis=1),
                                    axis=0)
        control2[stage] = np.nanargmin(np.nanmax(cost2[stage][states_played[stage]] + V_expanded2[states_played[stage]], axis=0),
                                    axis=0)

        states_played[stage + 1] = dynamics[stage, states_played[stage], control1[stage], control2[stage]]

    return control1, control2, states_played

if __name__ == '__main__':
    stage_count = 1
    dynamics = np.array([
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[2, 3], [1, 2]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]]
        ],
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[1, 2], [0, 3]],
            [[2, 3], [1, 2]],
            [[3, 4], [2, 3]],
            [[np.nan, np.nan], [np.nan, np.nan]]
        ]])

    costs1 = np.array([
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[0, 1], [-1, 0]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]]
        ],
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[-1, 0], [-2, 1]],
            [[0, 1], [-1, 0]],
            [[1, 2], [0, 1]],
            [[np.nan, np.nan], [np.nan, np.nan]]
        ]])

    costs2 = np.array([
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[0, -1], [1, 0]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]]
        ],
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[1, 0], [2, -1]],
            [[0, -1], [1, 0]],
            [[-1, -2], [0, -1]],
            [[np.nan, np.nan], [np.nan, np.nan]]
        ]])

    ctg1, ctg2 = generate_cost_to_go(stage_count, costs1, costs2)
    for k in range(stage_count + 2):
        print("V1[{}] = {}".format(k, ctg1[k]))
    for k in range(stage_count + 2):
        print("V2[{}] = {}".format(k, ctg2[k]))

    u, d, states = optimal_actions(stage_count, costs1, ctg1, costs2, ctg2, dynamics, 2)
    print('u =', u)
    print('d =', d)
    print('x =', states)
