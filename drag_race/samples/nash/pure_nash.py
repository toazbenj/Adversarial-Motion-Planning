import numpy as np

import numpy as np


def pure_nash_equilibrium_security(A, B):
    """
    Find all pure Nash equilibria for a bimatrix game using security policies (both players are minimizers).

    Parameters:
    A (2D np.array): Payoff matrix for Player 1 (row player)
    B (2D np.array): Payoff matrix for Player 2 (column player)

    Returns:
    List of tuples: Each tuple represents a pure Nash equilibrium (Player 1's strategy, Player 2's strategy)
    """
    # Convert matrices to numpy arrays if not already
    A = np.array(A)
    B = np.array(B)

    # Get number of strategies for both players
    m, n = A.shape

    # Step 1: Calculate Player 1's security level (maximin strategy for Player 1)
    player1_min_payoffs = np.max(A, axis=1)  # Minimize each row in matrix A
    player1_security_policy = np.argmin(player1_min_payoffs)  # Player 1 maximizes their minimum

    # Step 2: Calculate Player 2's security level (maximin strategy for Player 2)
    player2_min_payoffs = np.max(B, axis=0)  # Minimize each column in matrix B
    player2_security_policy = np.argmin(player2_min_payoffs)  # Player 2 maximizes their minimum

    value1 = A[player1_security_policy, player2_security_policy]
    value2 = B[player1_security_policy, player2_security_policy]

    return player1_security_policy, player2_security_policy, value1, value2


# Example payoff matrices for Player 1 and Player 2 (both minimizers)
A = np.array([[3, 6, 1],
              [8, -1, 5],
              [10, -2, 10]])

B = np.array([[-2, 5, -1],
              [10, 1, 5],
              [10, 2, 8]])
# A = np.array([
#     [2, 30],
#     [0, 8] ])
# B = np.array([
#     [2, 0],
#     [30, 8]])
# A = np.array([[4, 5],
#                [1, 3]])
# B = np.array([[1, 2],
#                [4, 5]])
# A = np.array([[2.45, 3.95],
#                [2.55, 4.05]])
# B = np.array([[1, 2],
#                [4, 5]])
# A = np.array([
#     [0, 0, 0, 0],
#     [2, 10, 2, 2],
#     [1, 1, 1, 1],
#     [3, 10, 3, 10]])
# B = np.array([
#     [1, 3, 0, 2],
#     [1, 10, 0, 2],
#     [1, 3, 0, 2],
#     [1, 10, 0, 10]])

# A = np.array([[-2, 1],
#               [0, -1]])
# B = np.array([[-3, 1],
#               [2, -2]])
# A = np.array([[2, 1, 3],
#               [2, 4, 3],
#               [5, 4, 6]])
# B = np.array([[6, 5, 4],
#               [3, 4, 2],
#               [2, 1, 3]])
# A = np.array([[3, 1],
#               [2, 0]])
#
# B = np.array([[2, 4],
#               [0, 1]])

# Find pure Nash equilibria
player1_security_policy, player2_security_policy, value1, value2 = pure_nash_equilibrium_security(A, B)
print("P1 NE Policy: ", player1_security_policy)
print("P2 NE Policy: ", player2_security_policy)
print("P1 Value: ", value1)
print("P2 Value: ", value2)

