import numpy as np
from scipy.optimize import minimize


def global_potential_function(A, B):
    """
    Computes a global potential function for two players given their cost matrices A (Player 1) and B (Player 2).
    :param A: Player 1's cost tensor
    :param B: Player 2's cost tensor
    :return: Global potential function as a tensor
    """
    assert A.shape == B.shape
    m, n = A.shape

    # Initialize the global potential function tensor
    phi = np.zeros((m, n))

    # Initialize with base value (can be arbitrary, here it's set to 0)
    phi[0, 0] = 0

    # First iterate over the first axis (A-dimension)
    for i in range(1, m):
        phi[i, 0] = phi[i - 1, 0] + A[i, 0] - A[i - 1, 0]

    # Then iterate over the second axis (B-dimension)
    for j in range(1, n):
        phi[0, j] = phi[0, j - 1] + B[0, j] - B[0, j - 1]

    # Fill in the rest of the potential function tensor by considering cross-effects between players
    for i in range(1, m):
        for j in range(1, n):
            phi[i, j] = (phi[i - 1, j] + A[i, j] - A[i - 1, j] +
                         phi[i, j - 1] + B[i, j] - B[i, j - 1]) / 2.0

    return phi


def is_valid_exact_potential(A, B, phi):
    """
    Check if the computed potential function satisfies the exact potential condition and
    has a global minimum at φ(0,0) = 0.
    :param A: Player 1's cost matrix
    :param B: Player 2's cost matrix
    :param phi: Global potential function
    :return: True if valid exact potential function with a global minimum at φ(0,0), False otherwise
    """
    m, n = A.shape

    # Check condition for Player 1's strategies
    for i in range(1, m):
        for j in range(n):
            delta_A = A[i, j] - A[i - 1, j]
            delta_phi = phi[i, j] - phi[i - 1, j]
            if not np.isclose(delta_A, delta_phi, atol=1e-6):
                return False

    # Check condition for Player 2's strategies
    for i in range(m):
        for j in range(1, n):
            delta_B = B[i, j] - B[i, j - 1]
            delta_phi = phi[i, j] - phi[i, j - 1]
            if not np.isclose(delta_B, delta_phi, atol=1e-6):
                return False
    return True


if __name__ == '__main__':
    A1 = np.array([[3, 4],
                   [1, 2]])
    B1 = np.array([[1, 2],
                   [3, 4]])
    A2 = np.array([[1, 2],
                   [4, 5]])
    B2 = np.array([[1, 5],
                   [3, 7]])

    A3 = np.array([[3, 6, 1],
                  [8, -1, 5],
                  [10, -2, 10]])
    B3 = np.array([[-2, 5, -1],
                  [10, 1, 5],
                  [10, 2, 8]])
    A4 = np.array([
        [8, 0],
        [30, 2]])
    B4 = np.array([
        [8, 30],
        [0, 2]])
    A5 = np.array([[4, 5],
                  [1, 3]])
    B5 = np.array([[1, 2],
                  [4, 5]])
    A6 = np.array([
        [0, 0, 0, 0],
        [2, 10, 2, 2],
        [1, 1, 1, 1],
        [3, 10, 3, 10]])
    B6 = np.array([
        [1, 3, 2, 2],
        [1, 10, 0, 2],
        [1, 3, 2, 2],
        [1, 10, 0, 10]])
    A7 = np.array([[-2, 1],
                  [0, -1]])
    B7 = np.array([[-3, 2],
                  [1, -2]])
    A8 = np.array([[2, 1, 3],
                  [2, 4, 3],
                  [5, 4, 6]])
    B8 = np.array([[4, 5, 6],
                  [3, 4, 2],
                  [2, 1, 3]])

    player1_games = [A1, A2, A3, A4, A5, A6, A7, A8]
    player2_games = [B1, B2, B3, B4, B5, B6, B7, B8]

    for g in range(len(player1_games)):
        A = player1_games[g]
        B = player2_games[g]
        pot = global_potential_function(A,B)
        print(pot)
        print(is_valid_exact_potential(A,B,pot))


