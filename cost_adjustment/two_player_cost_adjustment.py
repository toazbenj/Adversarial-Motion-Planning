import numpy as np
from scipy.optimize import minimize

def cost_adjustment(player1_games, player2_games):
    """
    Given lists of 2D cost tensors for Player 1 and Player 2, compute the error tensor such that it can be added to
    Player 1's cost tensor and produce an exact potential function for both players.
    The global potential function will have a unique minimum at phi[0, 0] = 0, with all other values > 0.
    Player 2's costs remain fixed.
    :param player1_games: list of 2D numpy arrays for Player 1
    :param player2_games: list of 2D numpy arrays for Player 2
    :return: list of error tensors for Player 1
    """

    player1_errors = []

    for i in range(len(player1_games)):
        A = player1_games[i]
        B = player2_games[i]

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
        result = minimize(objective, E_initial, constraints=constraints, method='trust-constr', options={'maxiter': 1000})

        # Debugging output to check if minimization is exiting too early
        print("Optimization Result:")
        print("Status:", result.status)  # 0 indicates successful optimization
        print("Message:", result.message)  # Check if there's any issue with optimization
        print("Number of Iterations:", result.nit)  # Ensure enough iterations are being performed
        print("Final Objective Value:", result.fun)  # Check the final value of the objective function

        # Extract the optimized error tensor for Player 1
        Ea_opt = result.x.reshape(A.shape)

        player1_errors.append(Ea_opt)

    return player1_errors


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
    # 4,6,9 currently failing
    A1 = np.array([[4, 5],
                   [1, 3]])
    A2 = np.array([[1, 2],
                   [4, 5]])
    B1 = np.array([[1, 2],
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
        [2, 30],
        [0, 8]])
    B4 = np.array([
        [2, 0],
        [30, 8]])

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
        [1, 3, 0, 2],
        [1, 10, 0, 2],
        [1, 3, 0, 2],
        [1, 10, 0, 10]])

    A7 = np.array([[-2, 1],
                  [0, -1]])
    B7 = np.array([[-3, 1],
                  [2, -2]])

    A8 = np.array([[2, 1, 3],
                  [2, 4, 3],
                  [5, 4, 6]])
    B8 = np.array([[6, 5, 4],
                  [3, 4, 2],
                  [2, 1, 3]])

    player1_games = [A1, A2, A3, A4, A5, A6, A7, A8]
    player2_games = [B1, B2, B3, B4, B5, B6, B7, B8]

    # Compute error tensors for Player 1
    player1_errors = cost_adjustment(player1_games, player2_games)

    # Add errors to Player 1's original costs
    player1_adjusted_costs = add_errors(player1_errors, player1_games)

    # Compute global potential functions based on adjusted costs
    potential_functions = []
    for i in range(len(player1_adjusted_costs)):
        potential = global_potential_function(player1_adjusted_costs[i], player2_games[i])
        potential_functions.append(potential)

    # Output the error tensors and potential functions
    output = {
        "player1_errors": player1_errors,
        "potential_functions": potential_functions
    }

    # Formatting the output for better readability
    for i, (p1_err, phi) in enumerate(zip(output['player1_errors'], output['potential_functions'])):
        print(f"Subgame {i+1}:\n")
        print(f"Player 1 Error Tensor:\n{p1_err}\n")
        print(f"Global Potential Function:\n{phi}\n")
        print("="*40, "\n")
