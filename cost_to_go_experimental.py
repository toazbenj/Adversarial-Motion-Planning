import numpy as np
from cost_to_go import cost

def generate_states(k):
    """
    Return numpy row array, each entry is nX by nU by nD, index by x, u, d, for cost
    :return:
    """
    # generate all possible states
    player_state_lst = []
    for distance in range(0, k+1):
        for lane in range(0, 2):
            for velocity in range(0, 2):
                state = [distance, lane, velocity]
                player_state_lst.append(state)

    state_lst = []
    for player_state1 in player_state_lst:
        for player_state2 in player_state_lst:
            state = np.array([player_state1, player_state2])
            state_lst.append(state.T)

    return state_lst



def generate_costs():
    pass
def generate_dynamics():
    pass


def cost_to_go(k, cost, dynamics):
    # Initialize V with zeros
    V = [None] * (k + 1)
    V[k] = np.zeros((cost[k - 1].shape[0], 1))

    # Iterate backwards from k to 1
    for k in range(k, 0, -1):
        V_next = V[k]
        cost_current = cost[k - 1]
        dynamics_current = dynamics[k - 1]

        # Calculate Vminmax and Vmaxmin
        Vminmax = np.min(np.max(cost_current + V_next[dynamics_current], axis=2), axis=1)
        Vmaxmin = np.max(np.min(cost_current + V_next[dynamics_current], axis=1), axis=2)

        # Check if saddle-point can be found
        if not np.array_equal(Vminmax, Vmaxmin):
            raise ValueError('Saddle-point cannot be found')

        # Assign Vminmax to V[k-1]
        V[k - 1] = Vminmax

    # If needed, convert V to desired format or print the result
    print("Round ", k, ": ", V)


if __name__ == '__main__':
    # state = np.array([[0, 0],
    #                   [0, 1],
    #                   [0, 0]])
    # control_input = np.array([[0, 0],
    #                           [0, 0]])
    # # maintaining speed, deceleration, acceleration,  turns, and collisions
    # penalty_lst = [0, 1, 2, 1, 10]
    #
    # G_cost_mat = generate_costs(state, control_input, penalty_lst)
    # F_dynamics_mat = generate_dynamics()
    # k = 2
    #
    # cost_to_go(k, G_cost_mat, F_dynamics_mat)

    states = generate_states(2)
    print(states)

# import numpy as np
#
# # Define the grid size
#
#
# # Update cost-to-go values
# for k in range(n * m):
#     for i in range(n):
#         for j in range(m):
#             for move in movements:
#                 ni, nj = i + move[0], j + move[1]
#                 if 0 <= ni < n and 0 <= nj < m:
#                     cost_to_go[i, j] = min(cost_to_go[i, j], 1 + cost_to_go[ni, nj])
#
# print("Cost-to-go matrix:")
# print(cost_to_go)
#
#
# if __name__ == '__main__':
#     # cost = np.array([])
#     # dynamics = np.array([])
#     # k = 2
#     #
#     # cost_to_go(k, cost, dynamics)
#
#     n, m = 3, 3
#
#     # Initialize cost-to-go matrix with high values
#     cost_to_go = np.full((n, m), np.inf)
#     cost_to_go[n - 1, m - 1] = 0  # The cost at the goal is 0
#
#     # Define possible movements (down, up, right, left)
#     movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# import numpy as np
#
# # Define the grid size
# n, m = 3, 3
#
# # Initialize cost-to-go matrices with high values
# cost_to_go_min = np.full((n, m, n, m), np.inf)
# cost_to_go_max = np.full((n, m, n, m), -np.inf)
#
# # Possible movements (down, up, right, left)
# movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
#
# # Function to compute Manhattan distance between two points
# def manhattan_distance(p1, p2):
#     return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
#
# # Initialize the goal state costs (distance 0 when at the same cell)
# for i in range(n):
#     for j in range(m):
#         cost_to_go_min[i, j, i, j] = 0
#         cost_to_go_max[i, j, i, j] = 0
#
# # Update cost-to-go values iteratively
# for k in range(n * m):
#     for i1 in range(n):
#         for j1 in range(m):
#             for i2 in range(n):
#                 for j2 in range(m):
#                     if (i1, j1) != (i2, j2):  # Players are not on the same cell
#                         min_costs = []
#                         max_costs = []
#                         for move1 in movements:
#                             ni1, nj1 = i1 + move1[0], j1 + move1[1]
#                             if 0 <= ni1 < n and 0 <= nj1 < m:
#                                 for move2 in movements:
#                                     ni2, nj2 = i2 + move2[0], j2 + move2[1]
#                                     if 0 <= ni2 < n and 0 <= nj2 < m:
#                                         min_costs.append(1 + cost_to_go_min[ni1, nj1, ni2, nj2])
#                                         max_costs.append(1 + cost_to_go_max[ni1, nj1, ni2, nj2])
#                         if min_costs:
#                             cost_to_go_min[i1, j1, i2, j2] = min(min_costs)
#                         if max_costs:
#                             cost_to_go_max[i1, j1, i2, j2] = max(max_costs)
#
# # Output the cost-to-go matrices
# print("Cost-to-go matrix for Player 1 (minimizing distance):")
# print(cost_to_go_min[:, :, 0, 0])  # Example slice for visualization
# print("Cost-to-go matrix for Player 2 (maximizing distance):")
# print(cost_to_go_max[:, :, 0, 0])  # Example slice for visualization


# import numpy as np
#
# # Define the grid size
# n, m = 4, 5
#
# # Initialize cost-to-go matrices with high values
# cost_to_go_min = np.full((n, m, n, m), np.inf)
# cost_to_go_max = np.full((n, m, n, m), -np.inf)
#
# # Possible movements (down, up, right, left)
# movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
#
# # Function to compute Manhattan distance between two points
# def manhattan_distance(p1, p2):
#     return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
#
# # Initialize the goal state costs (distance 0 when at the same cell)
# for i in range(n):
#     for j in range(m):
#         cost_to_go_min[i, j, i, j] = 0
#         cost_to_go_max[i, j, i, j] = 0
#
# # Update cost-to-go values iteratively
# for k in range(n * m):
#     for i1 in range(n):
#         for j1 in range(m):
#             for i2 in range(n):
#                 for j2 in range(m):
#                     if (i1, j1) != (i2, j2):  # Players are not on the same cell
#                         min_costs = []
#                         max_costs = []
#                         for move1 in movements:
#                             ni1, nj1 = i1 + move1[0], j1 + move1[1]
#                             if 0 <= ni1 < n and 0 <= nj1 < m:
#                                 for move2 in movements:
#                                     ni2, nj2 = i2 + move2[0], j2 + move2[1]
#                                     if 0 <= ni2 < n and 0 <= nj2 < m:
#                                         min_costs.append(1 + cost_to_go_min[ni1, nj1, ni2, nj2])
#                                         max_costs.append(1 + cost_to_go_max[ni1, nj1, ni2, nj2])
#                         if min_costs:
#                             cost_to_go_min[i1, j1, i2, j2] = min(min_costs)
#                         if max_costs:
#                             cost_to_go_max[i1, j1, i2, j2] = max(max_costs)
#
# # Initialize starting positions
# player1_pos = (0, 0)
# player2_pos = (3, 4)
#
# # Simulation function
# def play_game(player1_pos, player2_pos, cost_to_go_min, cost_to_go_max, movements):
#     steps = 0
#     max_steps = 10  # Set a limit to prevent infinite loops in case of errors
#     history = []
#
#     while steps < max_steps and player1_pos != player2_pos:
#         history.append((player1_pos, player2_pos))
#         print(f"Step {steps}: Player 1 at {player1_pos}, Player 2 at {player2_pos}")
#
#         # Player 1's move to minimize distance
#         min_distance = np.inf
#         best_move1 = player1_pos
#         for move in movements:
#             new_pos = (player1_pos[0] + move[0], player1_pos[1] + move[1])
#             if 0 <= new_pos[0] < n and 0 <= new_pos[1] < m:
#                 distance = cost_to_go_min[new_pos[0], new_pos[1], player2_pos[0], player2_pos[1]]
#                 if distance < min_distance:
#                     min_distance = distance
#                     best_move1 = new_pos
#
#         # Player 2's move to maximize distance
#         max_distance = -np.inf
#         best_move2 = player2_pos
#         for move in movements:
#             new_pos = (player2_pos[0] + move[0], player2_pos[1] + move[1])
#             if 0 <= new_pos[0] < n and 0 <= new_pos[1] < m:
#                 distance = cost_to_go_max[player1_pos[0], player1_pos[1], new_pos[0], new_pos[1]]
#                 if distance > max_distance:
#                     max_distance = distance
#                     best_move2 = new_pos
#
#         player1_pos = best_move1
#         player2_pos = best_move2
#         steps += 1
#
#     history.append((player1_pos, player2_pos))
#     print(f"Final Step {steps}: Player 1 at {player1_pos}, Player 2 at {player2_pos}")
#     return history
#
# # Run the simulation
# history = play_game(player1_pos, player2_pos, cost_to_go_min, cost_to_go_max, movements)
#
# # Output the history of the game
# print("History of the game:")
# for step, (pos1, pos2) in enumerate(history):
#     print(f"Step {step}: Player 1 at {pos1}, Player 2 at {pos2}")
