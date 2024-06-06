# https://python-control.readthedocs.io/en/latest/generated/control.dlqr.html

import numpy as np
import control

# Define the system matrices for a discrete-time linear system
A = np.array([[1.2, 0.5],
              [0, 0.8]]) # state matrix
B = np.array([[0.],
              [1.]])  # state update
Q = np.array([[1, 0],
              [0, 1]])  # State cost matrix
R = np.array([[0.1]])   # Control cost matrix

# Solve the discrete-time LQR problem
K, S, E = control.dlqr(A, B, Q, R)

print("Optimal feedback gain K:")
print(K)
print("\nSolution to the Riccati equation S:")
print(S)
print("\nEigenvalues of the closed-loop system E:")
print(E)

