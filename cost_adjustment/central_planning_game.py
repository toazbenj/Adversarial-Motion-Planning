import numpy as np
from scipy.optimize import minimize


def cost_adjustment(player1_games, player2_games, player3_games):
    """
    Given lists of 3D cost tensors for each player, compute the error tensor such that each error tensor can be added to
    the cost tensor and produce an exact potential function for all subgames involving all 3 players.
    The global potential function will have a unique minimum at phi[0, 0, 0] = 0, with all other values > 0.
    :param player1_games: list of 3D numpy arrays
    :param player2_games: list of 3D numpy arrays
    :param player3_games: list of 3D numpy arrays
    :return: lists of error tensors for each subgame for each player
    """

    player1_errors = []
    player2_errors = []
    player3_errors = []

    for i in range(len(player1_games)):
        A = player1_games[i]
        B = player2_games[i]
        C = player3_games[i]

        # Initialize error tensors for this game
        Ea = np.zeros_like(A)
        Eb = np.zeros_like(B)
        Ec = np.zeros_like(C)

        # Define the objective function
        def objective(E):
            Ea = E[:np.prod(A.shape)].reshape(A.shape)
            Eb = E[np.prod(A.shape):2 * np.prod(B.shape)].reshape(B.shape)
            Ec = E[2 * np.prod(B.shape):].reshape(C.shape)

            # Adjusted cost tensors
            A_prime = A + Ea
            B_prime = B + Eb
            C_prime = C + Ec

            # Compute the global potential function
            phi = global_potential_function(A_prime, B_prime, C_prime)

            # Regularization term: add small norm of the error tensors to avoid ill-conditioning
            regularization_term = 1e-6 * (np.linalg.norm(Ea) + np.linalg.norm(Eb) + np.linalg.norm(Ec))

            # Objective: norm of the potential function + regularization
            return np.linalg.norm(phi) + regularization_term

        def inequality_constraint(E):
            Ea = E[:np.prod(A.shape)].reshape(A.shape)
            Eb = E[np.prod(A.shape):2 * np.prod(B.shape)].reshape(B.shape)
            Ec = E[2 * np.prod(B.shape):].reshape(C.shape)

            # Adjusted cost tensors
            A_prime = A + Ea
            B_prime = B + Eb
            C_prime = C + Ec

            # Compute the global potential function
            phi = global_potential_function(A_prime, B_prime, C_prime)

            # Ensure all phi values (except phi[0, 0, 0]) are greater than a small positive epsilon
            epsilon = 1e-6
            return phi.flatten()[1:] - epsilon  # All values > epsilon instead of strictly > 0

        def constraint_phi_000(E):
            Ea = E[:np.prod(A.shape)].reshape(A.shape)
            Eb = E[np.prod(A.shape):2 * np.prod(B.shape)].reshape(B.shape)
            Ec = E[2 * np.prod(B.shape):].reshape(C.shape)

            # Adjusted cost tensors
            A_prime = A + Ea
            B_prime = B + Eb
            C_prime = C + Ec

            # Compute the global potential function
            phi = global_potential_function(A_prime, B_prime, C_prime)

            # Enforce phi[0, 0, 0] = 0
            return phi[0, 0, 0]


        # Flatten the initial error tensors
        E_initial = np.hstack([Ea.flatten(), Eb.flatten(), Ec.flatten()])

        # Set up the constraints
        constraints = [{'type': 'eq', 'fun': constraint_phi_000},
                       {'type': 'ineq', 'fun': inequality_constraint}]

        # Minimize the objective function (norm of the global potential function)
        result = minimize(objective, E_initial, constraints=constraints, method='SLSQP')

        # Debugging output to check if minimization is exiting too early
        print("Optimization Result:")
        print("Status:", result.status)  # 0 indicates successful optimization
        print("Message:", result.message)  # Check if there's any issue with optimization
        print("Number of Iterations:", result.nit)  # Ensure enough iterations are being performed
        print("Final Objective Value:", result.fun)  # Check the final value of the objective function

        # Extract the optimized error tensors
        Ea_opt = result.x[:np.prod(A.shape)].reshape(A.shape)
        Eb_opt = result.x[np.prod(A.shape):2 * np.prod(B.shape)].reshape(B.shape)
        Ec_opt = result.x[2 * np.prod(B.shape):].reshape(C.shape)

        player1_errors.append(Ea_opt)
        player2_errors.append(Eb_opt)
        player3_errors.append(Ec_opt)

    return player1_errors, player2_errors, player3_errors

def global_potential_function(A, B, C):
    """
    Computes a global potential function for all three players given their cost matrices A, B, and C.
    :param A: Player 1's cost tensor
    :param B: Player 2's cost tensor
    :param C: Player 3's cost tensor
    :return: Global potential function as a tensor
    """
    assert A.shape == B.shape == C.shape
    m, n, p = A.shape

    # Initialize the global potential function tensor
    phi = np.zeros((m, n, p))

    # Initialize with base value (can be arbitrary, here it's set to 0)
    phi[0, 0, 0] = 0

    # First iterate over the first axis (A-dimension)
    for i in range(1, m):
        phi[i, 0, 0] = phi[i - 1, 0, 0] + A[i, 0, 0] - A[i - 1, 0, 0]

    # Then iterate over the second axis (B-dimension)
    for j in range(1, n):
        phi[0, j, 0] = phi[0, j - 1, 0] + B[0, j, 0] - B[0, j - 1, 0]

    # Then iterate over the third axis (C-dimension)
    for k in range(1, p):
        phi[0, 0, k] = phi[0, 0, k - 1] + C[0, 0, k] - C[0, 0, k - 1]

    # Finally, fill in the rest of the potential function tensor by considering cross-effects between players
    for i in range(1, m):
        for j in range(1, n):
            for k in range(1, p):
                phi[i, j, k] = (phi[i - 1, j, k] + A[i, j, k] - A[i - 1, j, k] +
                                phi[i, j - 1, k] + B[i, j, k] - B[i, j - 1, k] +
                                phi[i, j, k - 1] + C[i, j, k] - C[i, j, k - 1]) / 3.0

    return phi


def add_errors(player1_errors, player2_errors, player3_errors, player1_games, player2_games, player3_games):
    """
    Add the computed error tensors to the original cost tensors for each player.
    """
    player1_adjusted = [player1_games[i] + player1_errors[i] for i in range(len(player1_games))]
    player2_adjusted = [player2_games[i] + player2_errors[i] for i in range(len(player2_games))]
    player3_adjusted = [player3_games[i] + player3_errors[i] for i in range(len(player3_games))]

    return player1_adjusted, player2_adjusted, player3_adjusted


if __name__ == '__main__':
    A1 = np.array([[[1, 2],
                    [4, 5]],
                   [[1, 2],
                    [3, 4]]])
    A2 = np.array([[[3, 4],
                    [2, 1]],
                   [[4, 2],
                    [1, 3]]])
    B1 = np.array([[[1, 2],
                    [4, 5]],
                   [[1, 2],
                    [3, 4]]])
    B2 = np.array([[[1, 5],
                    [3, 7]],
                   [[3, 7],
                    [5, 1]]])
    C1 = np.array([[[8, 6],
                    [4, 2]],
                   [[8, 2],
                    [6, 4]]])
    C2 = np.array([[[2, 8],
                    [4, 6]],
                   [[4, 6],
                    [2, 8]]])

    player1_games = [A1, A2]
    player2_games = [B1, B2]
    player3_games = [C1, C2]

    # Compute error tensors
    player1_errors, player2_errors, player3_errors = cost_adjustment(player1_games,
                                                                     player2_games,
                                                                     player3_games)

    # Add errors to original costs
    player1_adjusted_costs, player2_adjusted_costs, player3_adjusted_costs = add_errors(player1_errors,
                                                                                        player2_errors,
                                                                                        player3_errors,
                                                                                        player1_games,
                                                                                        player2_games,
                                                                                        player3_games)

    # Compute global potential functions based on adjusted costs
    potential_functions = []
    for i in range(len(player1_adjusted_costs)):
        potential = global_potential_function(player1_adjusted_costs[i], player2_adjusted_costs[i],
                                              player3_adjusted_costs[i])
        potential_functions.append(potential)

    # Output the error tensors and potential functions
    output = {
        "player1_errors": player1_errors,
        "player2_errors": player2_errors,
        "player3_errors": player3_errors,
        "potential_functions": potential_functions
    }

    # Formatting the output for better readability
    for i, (p1_err, p2_err, p3_err, phi) in enumerate(zip(output['player1_errors'], output['player2_errors'], output['player3_errors'], output['potential_functions'])):
        print(f"Subgame {i+1}:\n")
        print(f"Player 1 Error Tensor:\n{p1_err}\n")
        print(f"Player 2 Error Tensor:\n{p2_err}\n")
        print(f"Player 3 Error Tensor:\n{p3_err}\n")
        print(f"Global Potential Function:\n{phi}\n")
        print("="*40, "\n")