"""
Simple Zero Sum Game Solver
Created by Ben Toaz on 6-7-24

3 different algorithms to find mixed policies and game values, iterative algorithm is most consistent

# https://code.activestate.com/recipes/496825-game-theory-payoff-matrix-solver/
# https://chatgpt.com/share/4563dc68-176a-4577-862c-a9f8457a28ea

https://gambitproject.readthedocs.io/en/latest/pygambit.user.html#computing-nash-equilibria

https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.optimize.linprog.html
https://chatgpt.com/share/63683362-de69-4a3b-952c-a31e3afcdf3f
"""

import numpy as np
import pygambit as gbt
import numpy as np
from scipy.optimize import linprog

def find_admissible_ne(game, equilibra):
    value_of_game = 100000
    row_strategy = np.array([[],[]])
    col_strategy = np.array([[],[]])
    for eqm in equilibra:
        row_strat_object = list(eqm.__getitem__(game.players[0]))
        col_strat_object = list(eqm.__getitem__(game.players[1]))
        row_strat = [float(p[1]) for p in row_strat_object]
        col_strat = [float(p[1]) for p in col_strat_object]

        new_value = np.transpose(row_strat) @ payoff_matrix @ col_strat
        if new_value < value_of_game:
            value_of_game = new_value
            row_strategy = row_strat
            col_strategy = col_strat

    return row_strategy, col_strategy, value_of_game

def solve_with_gambit(payoff_matrix):
    game = gbt.Game.from_arrays(payoff_matrix, np.transpose(payoff_matrix))

    nash = gbt.nash.simpdiv_solve(game.mixed_strategy_profile(rational=True))
    # policies = game.mixed_strategy_profile(rational=True)
    # row_strategy = policies[game.players[0]]
    # col_strategy = policies[game.players[1]]
    # value_of_game = policies.payoff(game.players[1])

    result = gbt.nash.enummixed_solve(game, False, False)
    row_strategy, col_strategy, value_of_game = find_admissible_ne(game, result.equilibria)
    return row_strategy, col_strategy, value_of_game


def itr_solve(payoff_matrix, iterations=5000):
    'Return the oddments (mixed strategy ratios) for a given payoff matrix'
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


def scipy_solve(payoff_matrix):
    # Number of strategies for each player
    num_strategies_A, num_strategies_B = payoff_matrix.shape

    # Player A's problem: Maximize min expected payoff
    c_A = np.zeros(num_strategies_A + 1)
    c_A[-1] = -1  # Minimize the variable representing the value of the game

    # Constraints for Player A
    A_ub_A = np.zeros((num_strategies_B, num_strategies_A + 1))
    b_ub_A = np.zeros(num_strategies_B)

    for i in range(num_strategies_B):
        A_ub_A[i, :num_strategies_A] = -payoff_matrix[:, i]
        A_ub_A[i, -1] = 1

    A_eq_A = np.ones((1, num_strategies_A + 1))
    A_eq_A[0, -1] = 0
    b_eq_A = np.array([1])

    bounds_A = [(0, 1)] * num_strategies_A + [(None, None)]

    # Solve Player A's linear programming problem
    result_A = linprog(c_A, A_ub=A_ub_A, b_ub=b_ub_A, A_eq=A_eq_A, b_eq=b_eq_A, bounds=bounds_A, method='highs')

    # Constraints for Player B
    c_B = np.zeros(num_strategies_B + 1)
    c_B[-1] = -1  # Minimize the variable representing the value of the game

    A_ub_B = np.zeros((num_strategies_A, num_strategies_B + 1))
    b_ub_B = np.zeros(num_strategies_A)

    for i in range(num_strategies_A):
        A_ub_B[i, :num_strategies_B] = -payoff_matrix[i, :]
        A_ub_B[i, -1] = 1

    A_eq_B = np.ones((1, num_strategies_B + 1))
    A_eq_B[0, -1] = 0
    b_eq_B = np.array([1])

    bounds_B = [(0, 1)] * num_strategies_B + [(None, None)]

    # Solve Player B's linear programming problem
    result_B = linprog(c_B, A_ub=A_ub_B, b_ub=b_ub_B, A_eq=A_eq_B, b_eq=b_eq_B, bounds=bounds_B, method='highs')

    return result_A.x[:num_strategies_A], result_B.x[:num_strategies_B], result_A.x[-1]


def show_results(row_strategy, col_strategy, value_of_game):
    # print("Row player's mixed strategy:", row_strategy.round(3))
    # print("Column player's mixed strategy:", col_strategy.round(3))
    # print("Value of the game:", round(value_of_game, 3), "\n")

    print("Row player's mixed strategy:", row_strategy)
    print("Column player's mixed strategy:", col_strategy)
    print("Value of the game:", round(value_of_game, 3), "\n")


if __name__ == "__main__":
    # payoff_matrix = np.array([[-2, 0],
    #                           [3, -1]])
    # row_strategy, col_strategy, value_of_game = itr_solve(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)

    # need to flip row player strat
    # row_strategy, col_strategy, value_of_game = scipy_solve(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)

    # row_strategy, col_strategy, value_of_game = solve_with_gambit(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)

    # payoff_matrix = np.array([
    #     [0, 1, -1],
    #     [-1, 0, 1],
    #     [1, -1, 0]])
    # row_strategy, col_strategy, value_of_game = itr_solve(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)

    # row_strategy, col_strategy, value_of_game = scipy_solve(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)

    # payoff_matrix = np.array([[1, 4],
    #                           [3, -1]])
    # row_strategy, col_strategy, value_of_game = itr_solve(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)
    #
    # # row_strategy, col_strategy, value_of_game = scipy_solve(payoff_matrix)
    # # show_results(row_strategy, col_strategy, value_of_game)
    #
    # payoff_matrix = np.array([[1, 3, 2],
    #                           [3, 2, -2]])
    # row_strategy, col_strategy, value_of_game = itr_solve(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)
    #
    # # row_strategy, col_strategy, value_of_game = scipy_solve(payoff_matrix)
    # # show_results(row_strategy, col_strategy, value_of_game)
    #
    # payoff_matrix = np.array([[0, 1, 2, 3],
    #                           [1, 0, 1, 2],
    #                           [0, 1, 0, 1],
    #                           [-1, 0, 1, 0]])
    payoff_matrix = np.array([[1, 2],
                              [3, 0],])
    row_strategy, col_strategy, value_of_game = itr_solve(payoff_matrix)
    show_results(row_strategy, col_strategy, value_of_game)
    #
    # # row_strategy, col_strategy, value_of_game = scipy_solve(payoff_matrix)
    # # show_results(row_strategy, col_strategy, value_of_game)
    #
    # # row_strategy, col_strategy, value_of_game = solve_with_gambit(payoff_matrix)
    # # show_results(row_strategy, col_strategy, value_of_game)
    #
    # payoff_matrix = np.array([[0, 1],
    #                           [-1, 0]])
    # row_strategy, col_strategy, value_of_game = itr_solve(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)
    #
    # row_strategy, col_strategy, value_of_game = scipy_solve(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)

    # print("\n", "Gambit")
    # payoff_matrix = np.array([
    #     [0, 1, -1],
    #     [-1, 0, 1],
    #     [1, -1, 0]])
    # row_strategy, col_strategy, value_of_game = solve_with_gambit(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)
    #
    # payoff_matrix = np.array([[1, 4],
    #                           [3, -1]])
    # row_strategy, col_strategy, value_of_game = solve_with_gambit(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)
    #
    # payoff_matrix = np.array([[3, 0],
    #                           [-1, 1]])
    # row_strategy, col_strategy, value_of_game = solve_with_gambit(payoff_matrix)
    # show_results(row_strategy, col_strategy, value_of_game)

