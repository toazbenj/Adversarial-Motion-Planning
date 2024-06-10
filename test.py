"""
Test.py
Created by Ben Toaz on 6-7-24

Additional testing with Gambit library (ultimately disappointing, correct NE but wrong row player mixed policy)
"""


import numpy as np
import pygambit as gbt

# # Define the payoff matrix
# payoff_matrix = np.array([[0,1,2,3],
#                           [1, 0, 1, 2],
#                           [0,1,0,1],
#                           [-1,0,1,0]])
#
# # Create the game from the payoff matrices
# game = gbt.Game.from_arrays(payoff_matrix, np.transpose(payoff_matrix))
#
# # Find Nash equilibria
# result = gbt.nash.enummixed_solve(game, rational=False)
#
# # Assuming you want to find the value for the first equilibrium
# equilibrium = result.equilibria[3]
#
# # Extract the mixed strategy probabilities
# player1_strategy = equilibrium[game.players[0]]
# player2_strategy = equilibrium[game.players[1]]
#
# # Convert mixed strategies to list of probabilities
# player1_probabilities = np.array([float(prob[1]) for prob in player1_strategy])
# player2_probabilities = np.array([float(prob[1]) for prob in player2_strategy])
#
# # Verify extracted probabilities
# print("Player 1 mixed strategy probabilities:", player1_probabilities)
# print("Player 2 mixed strategy probabilities:", player2_probabilities)
#
# # Calculate the expected payoff for Player 1
# expected_payoff_player1 = np.dot(player1_probabilities, np.dot(payoff_matrix, player2_probabilities))
#
# # Calculate the expected payoff for Player 2
# expected_payoff_player2 = np.dot(player2_probabilities, np.dot(payoff_matrix.T, player1_probabilities))
#
# print("Expected Payoff for Player 1:", expected_payoff_player1)
# print("Expected Payoff for Player 2:", expected_payoff_player2)


import numpy as np

# Example coordinates as numpy arrays
coord1 = np.array([10, 20])
coord2 = np.array([30, 40])

# Convert to tuples
coord1_tuple = tuple(coord1)
coord2_tuple = tuple(coord2)

# Create dictionary with tuple keys
coord_dict_tuple = {
    coord1_tuple: "Data at (10, 20)",
    coord2_tuple: "Data at (30, 40)"
}

print(coord_dict_tuple[coord1_tuple])  # Output: "Data at (10, 20)"

# Convert to bytes
coord1_bytes = coord1.tobytes()
coord2_bytes = coord2.tobytes()

# Create dictionary with bytes keys
coord_dict_bytes = {
    coord1_bytes: "Data at (10, 20)",
    coord2_bytes: "Data at (30, 40)"
}

print(coord_dict_bytes[coord1_bytes])  # Output: "Data at (10, 20)"
