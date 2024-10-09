"""
Utilities

Basic functions for doing game theory calculations, numpy array manipulation, and more common drag race game
helper functions. Most assume game is bimatrix, where each player is a minimizer.
"""

import numpy as np
from scipy.optimize import linprog
from drag_race.utils.upkeep_utils import clean_matrix, remap_values

def mixed_policy_2d(payoff_matrix, iterations=5000, is_min_max=True):
    """
    Calculate the mixed policies and values for each player, security policies for one player at a time since non-ego
    policy calculated is worst case for ego player, not best case for non-ego player (N-minimizers)
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
    """
    Checks if the current state is a terminal state
    :param state_idx: int current state index
    :param state_lst: list of state arrays
    :param stage_count: int number of decision epochs until the final state
    :return: bool if current state is terminal state
    """
    state = state_lst[state_idx]
    if state[0][0] == stage_count + 1:
        return True
    elif state[0][1] == stage_count + 1:
        return True
    else:
        return False


def mixed_policy_3d(total_cost, state_lst, stage_count, is_min_max=True):
    """
    Find mixed saddle point game value for every state (security policies)
    :param total_cost: 3D cost array of state x control input x control input
    :param state_lst: list of state arrays
    :param stage_count: int number of stages to play
    :param is_min_max: bool whether player 1 is minimizer or player 2
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

    return np.around(row_policy, 2), np.around(col_policy, 2), ctg


def bimatrix_mixed_policy(total_cost1, total_cost2, state_lst, stage_count):
    """
    Find mixed saddle point game value for every state (approximate mixed nash equilibrium)
    :param total_cost1: 3D cost array of state x control input x control input
    :param total_cost2: 3D cost array of state x control input x control input
    :param state_lst: list of state arrays
    :param stage_count: int number of stages to play
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
            pass
        else:
            cost1 = clean_matrix(total_cost1[state])
            cost2 = clean_matrix(total_cost2[state])
            small_row_policy, small_col_policy, ctg1[state], ctg2[state] = scipy_solve(cost1, cost2)

            row_policy[state] = remap_values(total_cost1[state], small_row_policy)
            col_policy[state] = remap_values(total_cost2[state], small_col_policy, is_row=False)
            print(state, col_policy[state])

    return np.around(row_policy, 2), np.around(col_policy, 2), ctg1, ctg2


def scipy_solve(A, B):
    """
    Quadratic program implementation for finding policies for 2 minimizers (approximate mixed nash equilibrium)
    :param A: player 1 cost array
    :param B: player 2 cost array
    :return: list of floats optimal policies x2, float payoffs x2
    """
    m, n = A.shape

    # Construct the objective function vector
    c = np.zeros(m + n + 2)
    c[-2:] = -1

    # Construct the inequality constraint matrix Ain and vector bin
    Ain_top = np.hstack((np.zeros((m, m)), -A, np.ones((m, 1)), np.zeros((m, 1))))
    Ain_bottom = np.hstack((-B.T, np.zeros((n, n + 1)), np.ones((n, 1))))
    Ain = np.vstack((Ain_top, Ain_bottom))
    b_in = np.zeros(m + n)

    # Construct the equality constraint matrix Aeq and vector beq
    Aeq = np.zeros((2, m + n + 2))
    Aeq[0, :m] = 1
    Aeq[1, m:m + n] = 1
    beq = np.array([1, 1])

    # Define the bounds for the variables
    bounds = [(0, 1)] * (m + n) + [(-np.inf, np.inf), (-np.inf, np.inf)]

    # Solve the linear program
    result = linprog(c, A_ub=Ain, b_ub=b_in, A_eq=Aeq, b_eq=beq, bounds=bounds)

    if result.success:
        x = result.x
        y = x[:m]
        z = x[m:m + n]
        p = x[m + n]
        q = x[m + n + 1]
        return y, z, p, q
    else:
        raise ValueError("Linear programming did not converge")


def generate_moderate_policies(aggressive_policy1, aggressive_policy2, conservative_policy1, conservative_policy2):
    """
    Average two policies to produce intermediate policies with (supposedly) intermediate payoffs, hot fix hack version
    :param aggressive_policy1: player 1 minimizing rank costs array
    :param aggressive_policy2: player 2 minimizing rank costs array
    :param conservative_policy1: player 1 minimizing safety costs array
    :param conservative_policy2: player 2 minimizing safety costs array
    :return: tuple of moderate policy arrays (state x control inputs)
    """

    y = (aggressive_policy1 + conservative_policy1)/2
    z = (aggressive_policy2 + conservative_policy2)/2
    return y, z