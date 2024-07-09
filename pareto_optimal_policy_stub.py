import numpy as np
from utilities import scipy_solve
# Function to solve the game using scipy linprog


# Payoff matrices for both games
A1 = np.array([[3, 1],
               [0, 2]])
B1 = np.array([[2, 0],
               [1, 3]])

A2 = np.array([[4, 1],
               [3, 2]])
B2 = np.array([[1, 4],
               [2, 3]])

# Solve for optimal strategies in Game 1
y1, z1, p1, q1 = scipy_solve(A1, B1)
print(f"Optimal mixed strategies for Game 1:\nPlayer A: {y1}\nPlayer B: {z1}\np1: {p1}, q1: {q1}")

# Solve for optimal strategies in Game 2
y2, z2, p2, q2 = scipy_solve(A2, B2)
print(f"Optimal mixed strategies for Game 2:\nPlayer A: {y2}\nPlayer B: {z2}\np2: {p2}, q2: {q2}")
