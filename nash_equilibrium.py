"""
Simple Zero Sum Game Solver

Iterative algorithm for finding game value and player policies,
converted for use with numpy arrays from original source

# https://code.activestate.com/recipes/496825-game-theory-payoff-matrix-solver/
# https://chatgpt.com/share/4563dc68-176a-4577-862c-a9f8457a28ea

https://gambitproject.readthedocs.io/en/latest/pygambit.user.html#computing-nash-equilibria
"""

import numpy as np
import pygambit as gbt

def solve(payoff_matrix, iterations=10000):
    'Return the oddments (mixed strategy ratios) for a given payoff matrix'
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
    return rowcnt / iterations, colcnt / iterations, value_of_game


def solve_with_gambit(payoff_matrix):
    game = gbt.Game.from_arrays(payoff_matrix, np.transpose(payoff_matrix))

    # policies = game.mixed_strategy_profile(rational=True)
    # row_strategy = policies[game.players[0]]
    # col_strategy = policies[game.players[1]]
    # value_of_game = policies.payoff(game.players[1])

    result = gbt.nash.enummixed_solve(game, False, False)
    eqm = result.equilibria.pop()
    row_strategy = eqm[0]
    col_strategy = eqm[1]
    value_of_game = sum(row_strategy[i] * col_strategy[j] * payoff_matrix[i][j]
                        for i in range(len(row_strategy))
                        for j in range(len(col_strategy)))


    return row_strategy, col_strategy, value_of_game


def show_results(row_strategy, col_strategy, value_of_game):
    # print("Row player's mixed strategy:", row_strategy.round(3))
    # print("Column player's mixed strategy:", col_strategy.round(3))
    # print("Value of the game:", round(value_of_game, 3), "\n")

    print("Row player's mixed strategy:", row_strategy)
    print("Column player's mixed strategy:", col_strategy)
    print("Value of the game:", round(value_of_game, 3), "\n")


if __name__ == "__main__":
    payoff_matrix = np.array([
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0]])
    row_strategy, col_strategy, value_of_game = solve(payoff_matrix)
    show_results(row_strategy, col_strategy, value_of_game)

    payoff_matrix = np.array([[1, 4],
                              [3, -1]])
    row_strategy, col_strategy, value_of_game = solve(payoff_matrix)
    show_results(row_strategy, col_strategy, value_of_game)

    print("\n", "Gambit")
    payoff_matrix = np.array([
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0]])
    row_strategy, col_strategy, value_of_game = solve_with_gambit(payoff_matrix)
    show_results(row_strategy, col_strategy, value_of_game)

    payoff_matrix = np.array([[1, 4],
                              [3, -1]])
    row_strategy, col_strategy, value_of_game = solve_with_gambit(payoff_matrix)
    show_results(row_strategy, col_strategy, value_of_game)

