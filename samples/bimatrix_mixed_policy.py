"""
Bimatrix Mixed Policy
Testing for probabilistic play with multiple payoff matrices
"""

import numpy as np

def mixed_policy_2d(payoff_matrix, iterations=5000, is_row_player=True):
    """
    Calculate the mixed policies and values for each player
    :param payoff_matrix: game matrix with cost info
    :param is_row_player: if player 1 is minimizer or player 2 boolean
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
        if is_row_player:
            active_col = np.argmin(col_cum_payoff)
        else:
            active_col = np.argmax(col_cum_payoff)

        # Update column count and cumulative payoffs
        colcnt[active_col] += 1
        row_cum_payoff += transpose[active_col]

        # Choose the row with the maximum cumulative payoff
        if is_row_player:
            active_row = np.argmax(row_cum_payoff)
        else:
            active_row = np.argmin(row_cum_payoff)

    value_of_game = (np.max(row_cum_payoff) + np.min(col_cum_payoff)) / 2.0 / iterations
    return rowcnt / iterations, colcnt / iterations, round(value_of_game, 2)


if __name__ == '__main__':
    # A = np.array([[-2, 1],
    #               [0, -1]])
    # B = np.array([[-3, 1],
    #               [2, -2]])
    A = np.array([[1, 0, 1],
                  [1, -1, 1],
                  [1, -1, 1]])
    B = np.array([[-1, 0, -1],
                  [1, 1, 0],
                  [1, 1, 1]])
    row_policiy, _, value_of_game1 = mixed_policy_2d(A)
    _, col_policiy, value_of_game2 = mixed_policy_2d(B, is_row_player=False)
    print(row_policiy, col_policiy, value_of_game1, value_of_game2)