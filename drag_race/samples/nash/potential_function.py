import numpy as np
import matplotlib.pyplot as plt


# def potnetial_function(A, B):
#     """
#     Computes the global potential function for two players given their cost matrices A (Player 1) and B (Player 2),
#     and checks if the potential function satisfies the exact potential condition.
#
#     :param A: Player 1's cost matrix
#     :param B: Player 2's cost matrix
#     :return: Global potential function as a tensor, and a boolean indicating if it satisfies the exact potential condition.
#     """
#     assert A.shape == B.shape
#     m, n = A.shape
#
#     # Initialize the global potential function tensor
#     phi = np.zeros((m, n))
#     phi[0, 0] = 0
#
#     # Flag to track if the exact potential condition is satisfied
#     exact_potential = True
#
#     # Compute first row (Player 2's moves)
#     for j in range(1, n):
#         phi[0, j] = phi[0, j - 1] + B[0, j] - B[0, j - 1]
#         # Check exact potential condition for Player 2
#         if not np.isclose(B[0, j] - B[0, j - 1], phi[0, j] - phi[0, j - 1], atol=1e-6):
#             exact_potential = False
#
#     # Compute first column (Player 1's moves)
#     for i in range(1, m):
#         phi[i, 0] = phi[i - 1, 0] + A[i, 0] - A[i - 1, 0]
#         # Check exact potential condition for Player 1
#         if not np.isclose(A[i, 0] - A[i - 1, 0], phi[i, 0] - phi[i - 1, 0], atol=1e-6):
#             exact_potential = False
#
#     # Compute the rest of the grid (cross-effects between both players)
#     for i in range(1, m):
#         for j in range(1, n):
#             phi[i, j] = phi[i - 1, j] + A[i, j] - A[i - 1, j]
#             # Check exact potential condition for Player 1
#             if not np.isclose(A[i, j] - A[i - 1, j], phi[i, j] - phi[i - 1, j], atol=1e-6):
#                 exact_potential = False
#
#     return phi, exact_potential
#
def potential_function(A, B):
    """
    Computes the global potential function for two players given their cost matrices A (Player 1) and B (Player 2),
    and checks if the potential function satisfies the exact potential condition across both players' strategies.

    :param A: Player 1's cost matrix
    :param B: Player 2's cost matrix
    :return: Global potential function as a tensor, and a boolean indicating if it satisfies the exact potential condition.
    """
    assert A.shape == B.shape
    m, n = A.shape

    # Initialize the global potential function tensor
    phi = np.zeros((m, n))
    phi[0, 0] = 0

    # Flag to track if the exact potential condition is satisfied
    exact_potential = True

    # Compute first row (Player 2's moves)
    for j in range(1, n):
        phi[0, j] = phi[0, j - 1] + B[0, j] - B[0, j - 1]
        # Check exact potential condition for Player 2
        if not np.isclose(B[0, j] - B[0, j - 1], phi[0, j] - phi[0, j - 1], atol=1e-6):
            exact_potential = False

    # Compute first column (Player 1's moves)
    for i in range(1, m):
        phi[i, 0] = phi[i - 1, 0] + A[i, 0] - A[i - 1, 0]
        # Check exact potential condition for Player 1
        if not np.isclose(A[i, 0] - A[i - 1, 0], phi[i, 0] - phi[i - 1, 0], atol=1e-6):
            exact_potential = False

    # Compute the rest of the grid (cross-effects between both players)
    for i in range(1, m):
        for j in range(1, n):
            # Compute potential from Player 1's and Player 2's perspectives
            phi[i, j] = (phi[i - 1, j] + A[i, j] - A[i - 1, j] +
                         phi[i, j - 1] + B[i, j] - B[i, j - 1]) / 2.0

            # Check exact potential condition for Player 1
            if not np.isclose(A[i, j] - A[i - 1, j], phi[i, j] - phi[i - 1, j], atol=1e-6):
                exact_potential = False

            # Check exact potential condition for Player 2
            if not np.isclose(B[i, j] - B[i, j - 1], phi[i, j] - phi[i, j - 1], atol=1e-6):
                exact_potential = False

    return phi, exact_potential

def find_nash_equilibria(phi):
    m, n = phi.shape
    equilibria = []
    for i in range(m):
        for j in range(n):
            neighbors = []
            if i > 0:
                neighbors.append(phi[i - 1, j])
            if i < m - 1:
                neighbors.append(phi[i + 1, j])
            if j > 0:
                neighbors.append(phi[i, j - 1])
            if j < n - 1:
                neighbors.append(phi[i, j + 1])

            if all(phi[i, j] <= neighbor for neighbor in neighbors):
                equilibria.append((i, j))
    return equilibria


def game_values(A, B, indices):
    payoffs = []
    for pair in indices:
        payoffs.append((A[pair], B[pair]))
    return payoffs


def find_lowest_potential_points(potential_matrix):
    """
    Find the points corresponding to the lowest values in the potential matrix.
    :param potential_matrix: The matrix of potential values.
    :return: List of indices of the lowest potential points.
    """
    min_value = np.min(potential_matrix)
    lowest_points = np.argwhere(potential_matrix == min_value)
    return [tuple(point) for point in lowest_points]


# def is_pareto_efficient(costs):
#     """
#     Find Pareto-efficient points.
#     :param costs: Array of (A_cost, B_cost) pairs.
#     :return: Boolean array indicating whether each point is Pareto-efficient.
#     """
#     is_efficient = np.ones(costs.shape[0], dtype=bool)
#     for i, cost in enumerate(costs):
#         is_efficient[i] = not np.any(np.all(costs <= cost, axis=1) & np.any(costs < cost, axis=1))
#     return is_efficient


# def plot_pareto_frontier(A, B, potential_matrix, lowest_potential_points):
#     """
#     Plot the outcome pairs (A_cost, B_cost) and highlight the Pareto frontier and lowest potential points.
#     :param A: Payoff matrix A (as costs for player A).
#     :param B: Payoff matrix B (as costs for player B).
#     :param potential_matrix: The matrix of potential values.
#     :param lowest_potential_points: Indices of points with the lowest potential values.
#     """
#     # Get all possible pairs (A_cost, B_cost)
#     m, n = A.shape
#     outcomes = [(A[i, j], B[i, j]) for i in range(m) for j in range(n)]
#     outcomes = np.array(outcomes)
#
#     # Identify Pareto-efficient points
#     pareto_efficient = is_pareto_efficient(outcomes)
#
#     # Extract the lowest potential point outcomes
#     lowest_potential_outcomes = [(A[i, j], B[i, j]) for i, j in lowest_potential_points]
#
#     # Plot all outcomes
#     plt.scatter(outcomes[:, 0], outcomes[:, 1], label="All Outcomes", color="blue", alpha=0.7)
#
#     # Highlight Pareto-efficient outcomes
#     plt.scatter(outcomes[pareto_efficient][:, 0], outcomes[pareto_efficient][:, 1],
#                 label="Pareto Frontier", color="red")
#
#     # Highlight outcomes corresponding to the lowest potential points
#     lowest_potential_outcomes = np.array(lowest_potential_outcomes)
#     if lowest_potential_outcomes.size > 0:
#         plt.scatter(lowest_potential_outcomes[:, 0], lowest_potential_outcomes[:, 1],
#                     label="Lowest Potential Points", color="green", marker="*", s=150)
#
#     # Labels and plot details
#     plt.xlabel("Player A Cost")
#     plt.ylabel("Player B Cost")
#     plt.title("Pareto Frontier and Lowest Potential Points in Cost Space")
#     plt.legend()
#     plt.show()


if __name__ == '__main__':
    # A = np.array([[3, 6, 1],
    #               [8, -1, 5],
    #               [10, -2, 10]])
    #
    # B = np.array([[-2, 5, -1],
    #               [10, 1, 5],
    #               [10, 2, 8]])
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
    #
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
    A = np.array([[3, 1],
                  [2, 0]])

    B = np.array([[2, 5],
                  [0, 1]])
    # Compute the potential matrix
    phi, is_exact = potential_function(A, B)
    print("Potential Function:")
    print(phi)
    print("Is Exact Potential Function:", is_exact)

    # # Find the points corresponding to the lowest values in the potential matrix
    # lowest_potential_points = find_lowest_potential_points(potential_matrix)
    # print("Lowest Potential Points (indices):", lowest_potential_points)
    #
    # # Plot Pareto frontier in cost space and highlight lowest potential points
    # plot_pareto_frontier(A, B, potential_matrix, lowest_potential_points)
