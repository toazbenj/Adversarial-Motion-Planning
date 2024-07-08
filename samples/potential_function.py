"""
Potential Function

Functions that construct potential matrix, find local minimums, and game values

https://chatgpt.com/share/1e264510-d4c9-42f1-812f-5f582c4d726d
"""

import numpy as np

def potential_function(A, B):
    """
    Calculate potential function matrix
    :param A: first player payoff matrix
    :param B: 2nd player payoff matrix
    :return: numpy array matrix of payoff differentials
    """
    # Ensure that the matrices A and B have the same dimensions
    assert A.shape == B.shape

    m, n = A.shape  # dimensions of the matrices

    # Initialize the potential function matrix with zeros
    phi = np.zeros((m, n))

    # Set phi[0,0] = 0
    phi[0, 0] = 0

    # Fill the first row of phi
    for j in range(1, n):
        phi[0, j] = phi[0, j - 1] + B[0, j] - B[0, j - 1]

    # Fill the first column of phi
    for i in range(1, m):
        phi[i, 0] = phi[i - 1, 0] + A[i, 0] - A[i - 1, 0]

    # Fill the rest of the matrix
    for i in range(1, m):
        for j in range(1, n):
            phi[i, j] = phi[i - 1, j] + A[i, j] - A[i - 1, j]
            # Alternatively, the above can be:
            # phi[i, j] = phi[i, j-1] + B[i, j] - B[i, j-1]

    return phi


def find_nash_equilibria(phi):
    """
    Find all local minima from a potential function
    :param phi: potential function numpy array
    :return: list of local minima tuples
    """
    m, n = phi.shape
    equilibria = np.min(phi)

    return equilibria


def game_values(A, B, indices):
    """
    Get payoffs for each player
    :param A: 1st player payoffs numpy array
    :param B: 2nd player payoffs numpy array
    :param indices: coordinates of player choices tuple
    :return: tuple of payoffs for each player
    """
    payoffs = []
    for pair in indices:
        payoffs.append((A[pair], B[pair]))
    return payoffs


def pick_global_minima(equilibria, values):
    """
    Select global minima from list of local minima
    :param equilibria: list of local minima indicies tuples
    :param values: list of player payoff tuples
    :return: list of player choice tuples, list of player payoff tuples
    """
    minima_total = min(values, key=lambda x: x[0] + x[1])
    minima_p1 = min(values, key=lambda x: x[0])
    minima_p2 = min(values, key=lambda x: x[1])

    # set collapses redundant minima, no repeats
    minima_set = {minima_p1, minima_p2, minima_total}

    # remove non admissible equilibria
    minima_set_copy = minima_set.copy()
    for minima1 in minima_set_copy:
        for minima2 in minima_set_copy:
            if minima1 == minima2:
                continue
            elif minima1[0] >= minima2[0] and minima1[1] >= minima2[1] and minima1 in minima_set:
                minima_set.remove(minima1)

    # collect indicies of smallest values
    policies = []
    for minima in minima_set:
        policy = equilibria[values.index(minima)]
        policies.append(policy)

    return policies, minima_set



if __name__ == '__main__':
    # Example usage
    # A = np.array([
    #     [2, 30],
    #     [0, 8]
    # ])
    # B = np.array([
    #     [2, 0],
    #     [30, 8]
    # ])
    # A = np.array([
    #     [-2, 1],
    #     [0, -1]
    # ])
    # B = np.array([
    #     [-1, 3],
    #     [2, -2]
    # ])
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
    #
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
    # A = np.array([[1, 0, 1],
    #               [1, -1, 1],
    #               [1, -1, 1]])
    # B = np.array([[-1, 0, -1],
    #               [1, 1, 0],
    #               [1, 1, 1]])
    # A = np.array([[2, 0, 1],
    #               [2, -1, 0],
    #               [2, -2, 2]])
    # B = np.array([[-2, 0, -1],
    #               [2, 1, 0],
    #               [2, 2, 2]])
    # c = 10
    # A = np.array([[3, 0, 1],
    #               [c, -1, 0],
    #               [c, -2, c]])
    # B = np.array([[-2, 0, -1],
    #               [c, 1, 0],
    #               [c, 2, c]])
    A = 2*np.array([[3, 5, 1],
                  [10, -1, 5],
                  [10, -2, 10]])
    B = np.array([[-2, 5, -1],
                  [10, 1, 5],
                  [10, 2, 10]])
    potential_matrix = potential_function(A, B)
    print(potential_matrix)
    
    nash_equilibria = find_nash_equilibria(potential_matrix)
    print("Nash Equilibria indices:", nash_equilibria)
    
    nash_values = game_values(A, B, nash_equilibria)
    print("Nash values:", nash_values)
    
    policies, values = pick_global_minima(nash_equilibria, nash_values)
    print("Global policies: ", policies)
    print("Global values: ", values)