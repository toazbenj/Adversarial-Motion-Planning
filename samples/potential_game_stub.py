"""
Potential Game Stub

Functions for solving bimatrix games and finding mixed policies
"""
import numpy as np
from scipy.optimize import linprog, minimize

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

# Example usage
# A = np.array([[2, 30],
#               [0, 8]])
# B = np.array([[2, 0],
#               [30, 8]])
# A = np.array([[-2, 1],
#               [0, -1]])
# B = np.array([[-1, 2],
#               [3, -2]])
# c = 10
# A = np.array([[3, 0, 1],
#               [c, -1, 0],
#               [c, -2, c]])
# B = np.array([[-2, 0, -1],
#               [c, 1, 0],
#               [c, 2, c]])

# A = np.array([[3, 0, 1],
#               [10, -1, 0],
#               [10, -2, 10]])
# B = np.array([[-2, 0, -1],
#               [10, 1, 0],
#               [10, 4, 10]])
# A = np.array([[0, 1, -1],
#              [-1, 0, 1],
#              [1, -1, 0]])
# B = np.array([[0, -1, 1],
#              [1, 0, -1],
#              [-1, 1, 0]])
# A = np.array([[0, 1, 2, 3],
#               [1, 0, 1, 2],
#               [0, 1, 0, 1],
#               [-1, 0, 1, 0]])
A = np.array([[3, 5, 1],
              [10, -1, 5],
              [10, -2, 10]])
B = np.array([[-2, 5, -1],
              [10, 1, 5],
              [10, 2, 10]])

# A = np.array([[2, 4, 1, 2],
#               [10, 1, 5, 1],
#               [5, 1, 10, 2]])
# B = np.array([[-2, 4, -1, -2],
#               [10, -2, 5, -1],
#               [5, -1, 10, -2]])
# A = np.array([[10, 2, 5, 1],
#               [-2, 10, -1, 5],
#               [10, 1, 10, 2],
#               [-1, 10, -2, 10]])
# B = np.array([[10, -2, 5, -1],
#               [2, 10, 1, 5],
#               [10, -1, 10, -2],
#               [1, 10, 2, 10]])
player1_strategy, player2_strategy, value1, value2 = scipy_solve(A, B)
print("A=\n",A)
print("B=\n", B)
print(f"Player 1 policy: \n{player1_strategy}")
print(f"Player 2 policy: \n{player2_strategy}")
print("Game Value1: ", value1)
print("Game Value2: ", value2)
# print("Game Value: ", value)