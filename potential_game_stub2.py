import numpy as np
from scipy.optimize import minimize


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
# A = np.array([[1, -1],
#               [-1, 1]])
# B = np.array([[1, -1],
#               [-1, 1]])
A = np.array([[2, 30],
              [0, 8]])
B = np.array([[2, 0],
              [30, 8]])
s1_star, s2_star, value = solve_bimatrix_potential_game(A, B)

if s1_star is not None:
    print("Player 1's mixed strategy:", s1_star)
    print("Player 2's mixed strategy:", s2_star)
    print("Value of the game:", value)
else:
    print("No Nash equilibrium found.")
