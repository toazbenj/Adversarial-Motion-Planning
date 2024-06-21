"""
Potential Function

Functions that construct potential matrix, find local minimums, and game values

https://chatgpt.com/share/1e264510-d4c9-42f1-812f-5f582c4d726d
"""

import numpy as np

def calculate_potential_function_matrix(A, B):
    # Ensure that the matrices A and B have the same dimensions
    assert A.shape == B.shape

    m, n = A.shape  # dimensions of the matrices

    # Initialize the potential function matrix with zeros
    Phi = np.zeros((m, n))

    # Set Phi[0,0] = 0
    Phi[0, 0] = 0

    # Fill the first row of Phi
    for j in range(1, n):
        Phi[0, j] = Phi[0, j - 1] + B[0, j] - B[0, j - 1]

    # Fill the first column of Phi
    for i in range(1, m):
        Phi[i, 0] = Phi[i - 1, 0] + A[i, 0] - A[i - 1, 0]

    # Fill the rest of the matrix
    for i in range(1, m):
        for j in range(1, n):
            Phi[i, j] = Phi[i - 1, j] + A[i, j] - A[i - 1, j]
            # Alternatively, the above can be:
            # Phi[i, j] = Phi[i, j-1] + B[i, j] - B[i, j-1]

    return Phi


def find_nash_equilibria(Phi):
    m, n = Phi.shape
    equilibria = []

    # Iterate through each element in Phi
    for i in range(m):
        for j in range(n):
            # Determine neighbors
            neighbors = []
            if i > 0:
                neighbors.append(Phi[i - 1, j])  # Above
            if i < m - 1:
                neighbors.append(Phi[i + 1, j])  # Below
            if j > 0:
                neighbors.append(Phi[i, j - 1])  # Left
            if j < n - 1:
                neighbors.append(Phi[i, j + 1])  # Right

            # Check if Phi[i, j] is a local min
            if all(Phi[i, j] <= neighbor for neighbor in neighbors):
                equilibria.append((i, j))

    return equilibria


def game_values(A, B, indices):
    payoffs = []
    for pair in indices:
        payoffs.append((A[pair],B[pair]))
    return payoffs


# Example usage
# A = np.array([
#     [2, 30],
#     [0, 8]
# ])
# B = np.array([
#     [2, 0],
#     [30, 8]
# ])

A = np.array([
    [0, 0, 0, 0],
    [2, 10, 2, 2],
    [1, 1, 1, 1],
    [3, 10, 3, 10]])
B = np.array([
    [1, 3, 0, 2],
    [1, 10, 0, 2],
    [1, 3, 0, 2],
    [1, 10, 0, 10]])

potential_matrix = calculate_potential_function_matrix(A, B)
print(potential_matrix)

nash_equilibria = find_nash_equilibria(potential_matrix)
print("Nash Equilibria indices:", nash_equilibria)

nash_values = game_values(A, B, nash_equilibria)
print("Nash values:", nash_values)