"""https://nashpy.readthedocs.io/en/stable/"""

import numpy as np
import nashpy as nash

# Define the payoff matrices
# A = np.array([[3, 0, 1],
#               [10, -1, 0],
#               [10, -2, 10]])
# B = np.array([[-2, 0, -1],
#               [10, 1, 0],
#               [10, 2, 10]])

# A = np.array([[3, 5, 1],
#               [10, 5, 5],
#               [10, -2, 10]])
# B = np.array([[-2, 5, -1],
#               [10, 5, 5],
#               [10, 5, 10]])

A = np.array([[3, 5, 1],
                [10, -1, 5],
                [10, -2, 10]])
B = np.array([[-2, 5, -1],
              [10, 1, 5],
              [10, 2, 10]])

# A = np.array([[2, 30],
#               [0, 8]])
# B = np.array([[2, 0],
#               [30, 8]])
# A = np.array([[-2, 1],
#               [0, -1]])
# B = np.array([[-1, 2],
#               [3, -2]])
# Initialize the game
game = nash.Game(A, B)

# Compute mixed Nash equilibria
equilibria = list(game.support_enumeration(non_degenerate=False, tol=0))

policy1 = np.zeros(A.shape[0])
policy2 = np.zeros(A.shape[1])
payoff1 = 0
payoff2 = 0
for x, y in equilibria:
    print("Policies={}, {}".format(x, y, 2))
    u1 = x @ A @ y
    u2 = x @ B @ y
    print("Costs={}, {}".format(round(u1, 2), round(u2, 2)))
    if game.is_best_response(x, y) == (True, True):
        policy1 = x
        policy2 = y
        payoff1 = u1
        payoff2 = u2

print()
print("Admissible Policies={}, {}".format(policy1, policy2))
print("Equilibrium Costs={}, {}".format(payoff1, payoff2))
print("the end")