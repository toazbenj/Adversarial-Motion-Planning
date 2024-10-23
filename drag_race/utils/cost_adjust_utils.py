"""
Cost Adjustment Utilities

Functions for handling cost adjustment error minimization algorithm.
"""

import numpy as np
from scipy.optimize import minimize


def compute_column_norm(error_matrices):
    """
    Compute the first norm (1-norm) of each column for each error matrix.
    :param error_matrices: list of 2D numpy arrays (error tensors for Player 1)
    :return: list of 1D numpy arrays, each containing the column-wise 1-norms of the error matrix
    """
    max_column_norms_list = []

    for error_matrix in error_matrices:
        # Compute the column-wise 1-norm (Manhattan norm)
        column_norms = np.linalg.norm(error_matrix, ord=1, axis=0)
        # Get the maximum column norm
        max_column_norm = np.max(column_norms)
        max_column_norms_list.append(max_column_norm)

    return max_column_norms_list


def cost_adjustment(A, B):
    """
    Given lists of 2D cost tensors for Player 1 and Player 2, compute the error tensor such that it can be added to
    Player 1's cost tensor and produce an exact potential function for both players.
    The global potential function will have a unique minimum at phi[0, 0] = 0, with all other values > 0.
    Player 2's costs remain fixed.
    :param player1_games: list of 2D numpy arrays for Player 1
    :param player2_games: list of 2D numpy arrays for Player 2
    :return: list of error tensors for Player 1
    """

    # Initialize error tensor for Player 1
    Ea = np.zeros_like(A)

    # Define the objective function
    def objective(E):
        Ea = E.reshape(A.shape)

        # Adjusted cost tensors
        A_prime = A + Ea

        # Compute the global potential function
        phi = global_potential_function(A_prime, B)

        # Regularization term: add small norm of the error tensor to avoid ill-conditioning
        regularization_term = 1e-6 * np.linalg.norm(Ea)

        # Objective: norm of the potential function + regularization
        return np.linalg.norm(phi) + regularization_term

    def inequality_constraint(E):
        Ea = E.reshape(A.shape)

        # Adjusted cost tensors
        A_prime = A + Ea

        # Compute the global potential function
        phi = global_potential_function(A_prime, B)

        # Ensure all phi values (except phi[0, 0]) are greater than a small positive epsilon
        epsilon = 1e-6
        return phi.flatten()[1:] - epsilon  # All values > epsilon instead of strictly > 0

    def constraint_phi_00(E):
        Ea = E.reshape(A.shape)

        # Adjusted cost tensors
        A_prime = A + Ea

        # Compute the global potential function
        phi = global_potential_function(A_prime, B)

        # Enforce phi[0, 0] = 0
        return phi[0, 0]

    # Flatten the initial error tensor
    E_initial = Ea.flatten()

    # Set up the constraints
    constraints = [{'type': 'eq', 'fun': constraint_phi_00},
                   {'type': 'ineq', 'fun': inequality_constraint}]

    # Minimize the objective function (norm of the global potential function)
    # result = minimize(objective, E_initial, constraints=constraints, method='trust-constr', options={'maxiter': 1000})
    result = minimize(objective, E_initial, constraints=constraints, method='trust-constr', hess=None,
                      options={'maxiter': 1000})

    # Debugging output to check if minimization is exiting too early
    print("Optimization Result:")
    print("Status:", result.status)  # 0 indicates successful optimization
    print("Message:", result.message)  # Check if there's any issue with optimization
    print("Number of Iterations:", result.nit)  # Ensure enough iterations are being performed
    print("Final Objective Value:", result.fun)  # Check the final value of the objective function

    # Extract the optimized error tensor for Player 1
    Ea_opt = result.x.reshape(A.shape)

    return Ea_opt


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


def add_errors(player1_errors, player1_games):
    """
    Add the computed error tensors to the original cost tensors for Player 1.
    Player 2's costs remain unchanged.
    """
    player1_adjusted = [player1_games[i] + player1_errors[i] for i in range(len(player1_games))]
    return player1_adjusted


if __name__ == '__main__':
    B1 = np.array([[4, 5],
                   [1, 3]])
    B2 = np.array([[1, 2],
                   [4, 5]])

    player1_games = [B1]
    player2_games = [B2]

    # Compute error tensors for Player 1
    player1_errors = cost_adjustment(player1_games, player2_games)

    # Compute column-wise norms of the error tensors
    max_column_norms = compute_column_norm(player1_errors)

    # Add errors to Player 1's original costs
    player1_adjusted_costs = add_errors(player1_errors, player1_games)

    # Compute global potential functions based on adjusted costs
    potential_functions = []
    for i in range(len(player1_adjusted_costs)):
        potential = global_potential_function(player1_adjusted_costs[i], player2_games[i])
        potential_functions.append(potential)

    # Output the error tensors, potential functions, and maximum column-wise norms
    output = {
        "player1_errors": player1_errors,
        "potential_functions": potential_functions,
        "max_column_norms": max_column_norms
    }

    # Formatting the output for better readability
    for i, (p1_err, phi, max_col_norm) in enumerate(
            zip(output['player1_errors'], output['potential_functions'], output['max_column_norms'])):
        print(f"Subgame {i + 1}:\n")
        print(f"Player 1 Error Tensor:\n{p1_err}\n")
        print(f"Global Potential Function:\n{phi}\n")
        print(f"Maximum Column-wise 1-Norm of Error Tensor: {max_col_norm}\n")
        print("=" * 40, "\n")

    # Formatting the output for better readability
    for i, (p1_err, phi, max_col_norm) in enumerate(
            zip(output['player1_errors'], output['potential_functions'], output['max_column_norms'])):
        print(f"Subgame {i + 1}:\n")
        print(f"Player 1 Error Tensor:\n{p1_err}\n")
        print(f"Global Potential Function:\n{phi}\n")
        print(f"Maximum Column-wise 1-Norm of Error Tensor: {max_col_norm}\n")
        print("=" * 40, "\n")