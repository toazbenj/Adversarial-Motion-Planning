import numpy as np
from two_player_cost_adjustment import cost_adjustment, add_errors, global_potential_function, is_valid_exact_potential
# Define new test cases with 3x3 matrices for player1_games and player2_games

test_cases = [
    # Test Case 1: Matrices with uniform values
    (np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),

    # Test Case 2: Matrices with ascending values in rows
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])),

    # Test Case 3: Diagonal matrices (diagonal dominance)
    # fail
    (np.array([[10, 0, 0], [0, 5, 0], [0, 0, 1]]), np.array([[1, 0, 0], [0, 5, 0], [0, 0, 10]])),

    # Test Case 4: Negative values matrix
    (np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]), np.array([[-9, -8, -7], [-6, -5, -4], [-3, -2, -1]])),

    # Test Case 5: Random values with positive and negative entries
    # fail
    (np.array([[3, -4, 1], [7, -2, 5], [-10, 6, -3]]), np.array([[4, 5, -1], [3, -4, 2], [6, -7, 8]])),

    # Test Case 6: Sparse matrices with some zero entries
    (np.array([[0, 3, 0], [2, 0, 4], [0, 5, 0]]), np.array([[0, 1, 0], [3, 0, 2], [0, 4, 0]])),

    # Test Case 7: Symmetric matrices
    (np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]]), np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])),

    # Test Case 8: Matrices with increasing and decreasing rows and columns
    (np.array([[1, 3, 5], [7, 9, 11], [13, 15, 17]]), np.array([[17, 15, 13], [11, 9, 7], [5, 3, 1]])),

    # Test Case 9: Large values to test scaling
    # fail
    (np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]]),
     np.array([[900, 800, 700], [600, 500, 400], [300, 200, 100]])),

    # Test Case 10: All entries zero (edge case for potential functions)
    (np.zeros((3, 3)), np.zeros((3, 3)))
]

# Test runner to apply cost_adjustment and check results
for idx, (A, B) in enumerate(test_cases, 1):
    player1_games = [A]
    player2_games = [B]
    print(f"Running Test Case {idx}...\n")

    # Calculate error tensors and verify exact potential functions
    player1_errors = cost_adjustment(player1_games, player2_games)
    player1_adjusted_costs = add_errors(player1_errors, player1_games)

    # Compute global potential functions based on adjusted costs
    potential_function = global_potential_function(player1_adjusted_costs[0], player2_games[0])

    # Check if it's a valid exact potential
    is_valid = is_valid_exact_potential(player1_adjusted_costs[0], player2_games[0], potential_function)

    print(f"Test Case {idx} Result:")
    print(f"Player 1 Error Tensor:\n{player1_errors[0]}")
    print(f"Adjusted Player 1 Costs:\n{player1_adjusted_costs[0]}")
    print(f"Global Potential Function:\n{potential_function}")
    print(f"Is valid exact potential: {is_valid}")
    print("=" * 50 + "\n")
