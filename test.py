import numpy as np
import pygambit as gbt

# Define the payoff matrix
payoff_matrix = np.array([[0,1,2,3],
                          [1, 0, 1, 2],
                          [0,1,0,1],
                          [-1,0,1,0]])

# Create the game from the payoff matrices
game = gbt.Game.from_arrays(payoff_matrix, np.transpose(payoff_matrix))

# Find Nash equilibria
result = gbt.nash.enummixed_solve(game, rational=False)

# Assuming you want to find the value for the first equilibrium
equilibrium = result.equilibria[3]

# Extract the mixed strategy probabilities
player1_strategy = equilibrium[game.players[0]]
player2_strategy = equilibrium[game.players[1]]

# Convert mixed strategies to list of probabilities
player1_probabilities = np.array([float(prob[1]) for prob in player1_strategy])
player2_probabilities = np.array([float(prob[1]) for prob in player2_strategy])

# Verify extracted probabilities
print("Player 1 mixed strategy probabilities:", player1_probabilities)
print("Player 2 mixed strategy probabilities:", player2_probabilities)

# Calculate the expected payoff for Player 1
expected_payoff_player1 = np.dot(player1_probabilities, np.dot(payoff_matrix, player2_probabilities))

# Calculate the expected payoff for Player 2
expected_payoff_player2 = np.dot(player2_probabilities, np.dot(payoff_matrix.T, player1_probabilities))

print("Expected Payoff for Player 1:", expected_payoff_player1)
print("Expected Payoff for Player 2:", expected_payoff_player2)
