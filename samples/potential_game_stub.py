"""
Potential Game Stub

Functions for solving bimatrix games and finding mixed policies (not currently working)
"""



import numpy as np
from scipy.optimize import linprog


def scipy_solve(A, B):
    num_strategies_A = A.shape[0]
    num_strategies_B = A.shape[1]

    # Objective function for Player 1
    c1 = -A.sum(axis=1)

    # Constraints for Player 1
    A_ub1 = np.ones((1, num_strategies_A))
    b_ub1 = [1]

    # Bounds for the probabilities (between 0 and 1)
    bounds1 = [(0, 1)] * num_strategies_A

    # Solve the LP for Player 1
    res1 = linprog(c1, A_ub=A_ub1, b_ub=b_ub1, bounds=bounds1, method='highs')

    # Objective function for Player 2
    c2 = -B.sum(axis=0)

    # Constraints for Player 2
    A_ub2 = np.ones((1, num_strategies_B))
    b_ub2 = [1]

    # Bounds for the probabilities (between 0 and 1)
    bounds2 = [(0, 1)] * num_strategies_B

    # Solve the LP for Player 2
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs')

    # Get the mixed strategies
    player1_strategy = res1.x
    player2_strategy = res2.x

    return player1_strategy, player2_strategy, (res1.x[-1], res2.x[-1])


def itr_solve(A, B, iterations=5000):
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


import numpy as np


def lemke_howson(payoff1, payoff2):
    m, n = payoff1.shape
    A = np.hstack((np.eye(m), -payoff1))
    B = np.hstack((-payoff2.T, np.eye(n)))

    labels = [0] * (m + n)
    basis = [-1] * (m + n)
    for s in range(m + n):
        if s < m:
            labels[s] = -s - 1
        else:
            labels[s] = s - m + 1

    entering = -1
    idx = 0

    while entering != -1:
        idx += 1
        if idx % 2 == 1:
            entering = np.argmax(A[m:, 0])
            if A[m + entering, 0] <= 0:
                return None
        else:
            entering = np.argmax(B[:, n])
            if B[entering, n] <= 0:
                return None

        basis[entering] = labels[0]
        labels[0] = -entering - 1

        if entering < m:
            idx = np.argwhere(basis[:n] == entering)
            for v in range(n):
                if v != idx:
                    temp = A[:, v].copy()
                    A[:, v] = B[:, basis[v] + n].copy()
                    B[:, basis[v] + n] = temp.copy()
        else:
            idx = np.argwhere(basis[:n] == entering - m)
            for v in range(n):
                if v != idx:
                    temp = B[:, v].copy()
                    B[:, v] = A[:, basis[v]].copy()
                    A[:, basis[v]] = temp.copy()

    mixed_strategy1 = B[:n, n:m + n].dot(labels[:n])
    mixed_strategy2 = A[:m, m:n + m].dot(labels[n:m + n])

    value = mixed_strategy1.dot(payoff1).dot(mixed_strategy2)

    return mixed_strategy1, mixed_strategy2, value

def potential_function(s1, s2, A, B):
    return np.sum(np.dot(np.dot(s1, A), s2.T) * B)


def solve_bimatrix_potential_game(A, B):
    m, n = A.shape

    # Define the objective function for player 1
    def obj1(s1):
        return potential_function(s1, np.ones(n) / n, A, B)

    # Define the objective function for player 2
    def obj2(s2):
        return potential_function(np.ones(m) / m, s2, A, B)

    # Initial guesses for mixed strategies
    x0_player1 = np.ones(m) / m
    x0_player2 = np.ones(n) / n

    # Minimize potential function for player 1
    res_player1 = minimize(obj1, x0_player1, method='SLSQP', bounds=[(0, 1)] * m, options={'disp': False})

    # Minimize potential function for player 2
    res_player2 = minimize(obj2, x0_player2, method='SLSQP', bounds=[(0, 1)] * n, options={'disp': False})

    if res_player1.success and res_player2.success:
        s1_star = res_player1.x
        s2_star = res_player2.x
        value = potential_function(s1_star, s2_star, A, B)
        return s1_star, s2_star, value
    else:
        return None, None, None  # No equilibrium found


# Example usage
A = np.array([[2, 30],
              [0, 8]])
B = np.array([[2, 0],
              [30, 8]])
# player1_strategy, player2_strategy, value = scipy_solve(A, B)
player1_strategy, player2_strategy, value = lemke_howson(A, B)
print(f"Player 1 strategy: \n{player1_strategy}")
print(f"Player 2 strategy: \n{player2_strategy}")
print("Game Value: ", value)

# s1_star, s2_star, value = solve_bimatrix_potential_game(A, B)
#
# if s1_star is not None:
#     print("Player 1's mixed strategy:", s1_star)
#     print("Player 2's mixed strategy:", s2_star)
#     print("Value of the game:", value)
# else:
#     print("No Nash equilibrium found.")