import numpy as np
from scipy.optimize import minimize

import numpy as np
from scipy.optimize import minimize

def cost_adjustment(player1_games, player2_games):
    player1_errors = []

    for i in range(len(player1_games)):
        A = player1_games[i]
        B = player2_games[i]

        # Initialize error tensor for Player 1
        Ea = np.zeros_like(A)

        # Precompute the global potential function for the original matrices
        phi_initial = global_potential_function(A, B)

        # Check if the matrices already yield an exact potential function
        if is_valid_exact_potential(A, B, phi_initial):
            print(
                f"Subgame {i + 1}: Matrices already yield an exact potential function with φ(0,0) = 0. No adjustment needed.")
            player1_errors.append(Ea)  # Ea remains all zeros
            continue  # Skip the optimization step

        # Define the objective function to minimize the norm of the potential function
        def objective(E):
            Ea = E.reshape(A.shape)
            A_prime = A + Ea
            phi = global_potential_function(A_prime, B)
            regularization_term = 1e-6 * np.linalg.norm(Ea)
            return np.linalg.norm(phi) + regularization_term

        def inequality_constraint(E):
            Ea = E.reshape(A.shape)
            A_prime = A + Ea
            phi = global_potential_function(A_prime, B)
            epsilon = 1e-6
            return phi.flatten()[1:] - epsilon  # Ensure all phi values (except phi[0, 0]) > epsilon

        def constraint_phi_00(E):
            Ea = E.reshape(A.shape)
            A_prime = A + Ea
            phi = global_potential_function(A_prime, B)
            return phi[0, 0]  # Enforce φ[0, 0] = 0

        def constraint_exact_potential(E):
            """
            Enforce that for each increment in A and B, the increment in phi matches, creating an exact potential.
            """
            Ea = E.reshape(A.shape)
            A_prime = A + Ea
            phi = global_potential_function(A_prime, B)
            m, n = A.shape
            potential_diffs = []

            # Check exact potential conditions for Player 1's cost
            for i in range(1, m):
                for j in range(n):
                    delta_A = A_prime[i, j] - A_prime[i - 1, j]
                    delta_phi = phi[i, j] - phi[i - 1, j]
                    potential_diffs.append(delta_phi - delta_A)

            # Check exact potential conditions for Player 2's cost
            for i in range(m):
                for j in range(1, n):
                    delta_B = B[i, j] - B[i, j - 1]
                    delta_phi = phi[i, j] - phi[i, j - 1]
                    potential_diffs.append(delta_phi - delta_B)

            return np.array(potential_diffs)

        # Flatten the initial error tensor
        E_initial = Ea.flatten()

        # Set up the constraints
        constraints = [{'type': 'eq', 'fun': constraint_phi_00},
                       {'type': 'ineq', 'fun': inequality_constraint},
                       {'type': 'eq', 'fun': constraint_exact_potential}]

        # Minimize the objective function
        result = minimize(objective, E_initial, constraints=constraints, method='trust-constr', options={'maxiter': 1000})

        # Extract the optimized error tensor for Player 1
        Ea_opt = result.x.reshape(A.shape)
        player1_errors.append(Ea_opt)

    return player1_errors

# Rest of your helper functions remain unchanged

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


def is_valid_exact_potential(A, B, phi):
    """
    Check if the computed potential function satisfies the exact potential condition and
    has a global minimum at φ(0,0) = 0.
    :param A: Player 1's cost matrix
    :param B: Player 2's cost matrix
    :param phi: Global potential function
    :return: True if valid exact potential function with a global minimum at φ(0,0), False otherwise
    """
    m, n = A.shape

    # Check condition for Player 1's strategies
    for i in range(1, m):
        for j in range(n):
            delta_A = A[i, j] - A[i - 1, j]
            delta_phi = phi[i, j] - phi[i - 1, j]
            if not np.isclose(delta_A, delta_phi, atol=1e-6):
                return False

    # Check condition for Player 2's strategies
    for i in range(m):
        for j in range(1, n):
            delta_B = B[i, j] - B[i, j - 1]
            delta_phi = phi[i, j] - phi[i, j - 1]
            if not np.isclose(delta_B, delta_phi, atol=1e-6):
                return False

    # Check if φ(0,0) = 0
    if not np.isclose(phi[0, 0], 0, atol=1e-6):
        return False

    # Check if φ(0,0) is the global minimum
    if np.any(phi[1:] <= phi[0, 0]):
        return False

    return True


def add_errors(player1_errors, player1_games):
    """
    Add the computed error tensors to the original cost tensors for Player 1.
    Player 2's costs remain unchanged.
    """
    player1_adjusted = [player1_games[i] + player1_errors[i] for i in range(len(player1_games))]

    return player1_adjusted


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


if __name__ == '__main__':
    A1 = np.array([[3, 4],
                   [1, 2]])
    B1 = np.array([[1, 2],
                   [3, 4]])

    A2 = np.array([[1, 2],
                   [4, 5]])
    B2 = np.array([[1, 5],
                   [3, 7]])

    A3 = np.array([[3, 6, 1],
                  [8, -1, 5],
                  [10, -2, 10]])
    B3 = np.array([[-2, 5, -1],
                  [10, 1, 5],
                  [10, 2, 8]])

    A4 = np.array([
        [8, 0],
        [30, 2]])
    B4 = np.array([
        [8, 30],
        [0, 2]])

    A5 = np.array([[4, 5],
                  [1, 3]])
    B5 = np.array([[1, 2],
                  [4, 5]])

    A6 = np.array([
        [0, 0, 0, 0],
        [2, 10, 2, 2],
        [1, 1, 1, 1],
        [3, 10, 3, 10]])
    B6 = np.array([
        [1, 3, 2, 2],
        [1, 10, 0, 2],
        [1, 3, 2, 2],
        [1, 10, 0, 10]])

    A7 = np.array([[-2, 1],
                  [0, -1]])
    B7 = np.array([[-3, 2],
                  [1, -2]])

    A8 = np.array([[2, 1, 3],
                  [2, 4, 3],
                  [5, 4, 6]])
    B8 = np.array([[4, 5, 6],
                  [3, 4, 2],
                  [2, 1, 3]])

    player1_games = [A1, A2, A3, A4, A5, A6, A7, A8]
    player2_games = [B1, B2, B3, B4, B5, B6, B7, B8]

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

    for g in range(len(player1_games)):
        print(is_valid_exact_potential(player1_adjusted_costs[g],
                                       player2_games[g],
                                       potential_functions[g]))
