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


def cost_adjustment(A, B, global_min_position):
    """
    Given lists of 2D cost tensors for Player 1 and Player 2, compute the error tensor such that it can be added to
    Player 1's cost tensor and produce an exact potential function for both players.
    The global potential function will have a unique minimum at phi[0, 0] = 0, with all other values > 0.
    Player 2's costs remain fixed.
    :param player1_games: list of 2D numpy arrays for Player 1
    :param player2_games: list of 2D numpy arrays for Player 2
    :return: list of error tensors for Player 1
    """
    # phi_i = np.argmin(np.max(A2, axis=1), axis=0)
    # phi_j = np.argmin(np.max(B, axis=0), axis=0)
    phi_indicies = global_min_position

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
        flat_arr = phi.flatten()
        other_entries = np.delete(flat_arr, phi_indicies)

        return other_entries- epsilon  # All values > epsilon instead of strictly > 0

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
    Ea_opt = result.x.reshape(A1.shape)

    return Ea_opt + A


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


def global_potential_function_numeric(A, B, global_min_position):
    """Computes a global potential function for given cost matrices"""
    m, n = A.shape
    phi = np.zeros((m, n))

    for i in range(1, m):
        phi[i, 0] = phi[i - 1, 0] + A[i, 0] - A[i - 1, 0]

    for j in range(1, n):
        phi[0, j] = phi[0, j - 1] + B[0, j] - B[0, j - 1]

    for i in range(1, m):
        for j in range(1, n):
            phi[i, j] = (phi[i - 1, j] + A[i, j] - A[i - 1, j] +
                         phi[i, j - 1] + B[i, j] - B[i, j - 1]) / 2

    return phi - phi[global_min_position[0], global_min_position[1]]


def is_valid_exact_potential(A, B, phi, global_min_position):
    """Checks if the exact potential condition is met"""
    m, n = A.shape

    for i in range(1, m):
        for j in range(n):
            if abs((A[i, j] - A[i - 1, j]) - (phi[i, j] - phi[i - 1, j])) > 1e-6:
                return False

    for i in range(m):
        for j in range(1, n):
            if abs((B[i, j] - B[i, j - 1]) - (phi[i, j] - phi[i, j - 1])) > 1e-6:
                return False

    return True


def is_global_min_enforced(phi, global_min_position):
    """Checks if the global minimum is enforced"""
    m, n = phi.shape
    if phi[global_min_position[0], global_min_position[1]] != 0:
        return False

    for i in range(m):
        for j in range(n):
            if (i, j) != tuple(global_min_position) and phi[i, j] <= 0:
                return False

    return True


if __name__ == '__main__':
    # A1 = np.load(('green_scalar.npz'))['arr']
    # B1 = np.load(('blue_scalar.npz'))['arr'].transpose()
    size = 9
    A1 = np.random.uniform(0, 50, (size, size))
    B1 = np.random.uniform(0, 50, (size, size))

    # Compute potential functions for adjusted costs
    potential_functions = []

    player2_sec = np.argmin(np.max(B1, axis=0), axis=0)
    for k in range(A1.shape[0]):
        global_min_position = [k, player2_sec]
        E = cost_adjustment(A1, B1, global_min_position)

        print(k)

        if E is None:
            continue

        # Adjusted cost matrix for Player 1
        A_prime = A1 + E

        # Compute potential function for adjusted costs
        phi = global_potential_function_numeric(A_prime, B1, global_min_position)
        potential_functions.append(phi)

        # Validation check
        if is_global_min_enforced(phi, global_min_position) and phi[
            global_min_position[0], global_min_position[1]] == 0:
            print(f"Subgame {1} Results:")
            print("Player 1 A_prime:")
            print(A_prime)
            print("Potential Function:")
            print(phi)
            print("Global Min:", global_min_position)
            print("Global Minimum Enforced:", is_global_min_enforced(phi, global_min_position))
            print("Exact Potential:", phi[global_min_position[0], global_min_position[1]] == 0)
            print("----------------------------------------")
    print("==========================================================================")