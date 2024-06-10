"""
Cost_to_go_experimental.py
Created by Ben Toaz on 6-7-24

Small scale implementation of matlab algorithm described in chapter 17 of Hespanha's NONCOOPERATIVE GAME THEORY.
Hard coded values for 2 rounds, 2 players
"""

import numpy as np
from cost_to_go import cost

def collision_check(state, control_input):
    """
    Compares player positions and returns whether they are in the same position or collided during the next maneuver
    :param state: matrix of distance, lane, and velocity for each player
    :param control_input: matrix of lane change and acceleration for each player
    :return: boolean
    """

    # pos check
    if state[0][0][0] == state[0][0][1] and state[0][1][0] == state[0][1][1]:
        return True

    # maneuver check
    # vehicles not turning
    if control_input[0][0] == 0 and control_input[0][1] == 0:
        return False

    is_same_lane_change = (control_input[0][0]) == (-control_input[0][1])
    is_same_speed = (control_input[1][0] + state[0][2][0]) == (control_input[1][1] + state[0][2][1])
    is_same_distance = state[0][0][0] == state[0][0][1]
    if is_same_lane_change and is_same_speed and is_same_distance:
        return True

    return False


def safety_cost(state, control_input1, control_input2):
    """
    Compute the cost matrix associated with the current game state for each player

    :param state: matrix of distance, lane, and velocity for each player
    :param control_input: matrix of lane change and acceleration for each player
    :param penalty_lst: list of costs associated for maintaining speed, deceleration,
    acceleration,  turns, and collisions
    :return: cost matrix with for safety and rank objectives for each player
    """

    control_input_trans = np.array([control_input1.T, control_input2.T])
    control_input = control_input_trans.T
    penalty_lst = [0, 1, 2, 1, 10]

    # check for collisions

    # enforce boundary conditions for the track also
    collision_penalty_int = penalty_lst[-1]
    if collision_check(state, control_input1, control_input2):
        return np.array([[collision_penalty_int, collision_penalty_int],
                        [collision_penalty_int, collision_penalty_int]])
    else:
        # calculate rank
        ranks = [1, 1]
        p1_dist = state[0][0] + control_input1[0][1]
        p2_dist = state[0][1] + control_input2[0][1]
        if p1_dist > p2_dist:
            ranks = [0, 1]
        elif p1_dist < p2_dist:
            ranks = [1, 0]

        # calculate safety
        safety = [0, 0]
        for i in range(0, control_input.ndim-1):
            risk = 0
            # maintain speed
            if control_input[0][1][i] == 0:
                risk += penalty_lst[0]
            # decelerate
            elif control_input[0][1][i] < 0:
                risk += penalty_lst[1]
            # accelerate
            elif control_input[0][1][i] > 0:
                risk += penalty_lst[2]
            # turn
            if control_input[0][0][i] != 0:
                risk += penalty_lst[3]

            safety[i] = risk

        return np.array([safety,
                        ranks])

def rank_cost(state, control_input):
    """
    Compute the cost matrix associated with the current game state for each player

    :param state: matrix of distance, lane, and velocity for each player
    :param control_input: matrix of lane change and acceleration for each player
    :param penalty_lst: list of costs associated for maintaining speed, deceleration,
    acceleration,  turns, and collisions
    :return: single rank cost for both players, min and max with respect to player 1 position
    """

    penalty_lst = [0, 1, 2, 1, 10]

    # check for collisions
    # enforce boundary conditions for the track also
    collision_penalty_int = penalty_lst[-1]
    rank_cost = 0.5
    if collision_check(state, control_input):
        rank_cost = 0.5
    else:
        # calculate rank
        p1_dist = state[0][0][0] + control_input[0][0]
        p2_dist = state[0][0][1] + control_input[0][1]
        if p1_dist > p2_dist:
            rank_cost = 0
        elif p1_dist < p2_dist:
            rank_cost = 1

    return rank_cost

def generate_states(k):
    """
    Calculate all possible states over k rounds
    :param k: the number of stages in game
    :return: list of all possible numpy array states
    """
    # generate all possible states for a player
    player_state_lst = []
    for distance in range(0, k+1):
        for lane in range(0, 2):
            for velocity in range(0, 2):
                state = [distance, lane, velocity]
                player_state_lst.append(state)

    # make list of all combinations for both players
    state_lst = []
    for player_state1 in player_state_lst:
        for player_state2 in player_state_lst:
            state = np.array([player_state1, player_state2])
            state_lst.append(state.T)

    return state_lst

def generate_control_inputs():
    """
    Make list of all possible player actions given space/movement constraints
    :return: list of control input numpy arrays
    """
    control_input_lst = []
    lane_maneuver_range = range(-1, 2)
    acceleration_maneuver_range = range(-1, 2)
    for i in lane_maneuver_range:
        for j in acceleration_maneuver_range:
            for k in lane_maneuver_range:
                for l in acceleration_maneuver_range:
                    control_input_lst.append(np.array([[i, k],[j,l]]))

    return control_input_lst


def generate_costs(state_lst, control_input_lst):
    """
    Calculate stage cost given each state and control input
    :return: tensor of stage cost indexed by each state/control input
    """


    # fix cost dimensions
    cost_lookup_mat = np.zeros((len(state_lst), len(control_input_lst)))
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            cost_value = rank_cost(state_lst[i], control_input_lst[j])
            cost_lookup_mat[i, j] = cost_value

    return cost_lookup_mat

def dynamics(state, control_input):
    """
    Calculate next state from given current state and control input
    :param state: matrix of distance, lane, and velocity for each player
    :param control_input1: player one lane and velocity change matrix
    :param control_input2: player two lane and velocity change matrix
    :return: next state matrix
    """

    A = np.array([[1, 0, 1],
                  [0, 1, 0],
                  [0, 0, 1]])
    B = np.array([[0, 1],
                  [1, 0],
                  [0, 1]])
    next_state = np.dot(A, state) + np.dot(B, control_input)

    return next_state


def generate_dynamics(state_lst, control_input_lst):
    """
    Make lookup table for next state given current state and player actions, implemented as nested dictionary
    :param state_lst:
    :param control_input_lst:
    :return:
    """
    dynamics_lookup_mat = np.zeros((len(state_lst), len(control_input_lst), len(control_input_lst)), dtype=object)
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            for k in range(len(control_input_lst)):
                next_state_mat = dynamics(state_lst[i], control_input_lst[j], control_input_lst[k])
                dynamics_lookup_mat[i, j, k] = next_state_mat

    return dynamics_lookup_mat


def generate_cost_to_go(k, cost, dynamics):
    # Initialize V with zeros
    V = np.zeros((k+1, len(dynamics)))

    # Iterate backwards from k to 1
    for stage in range(k, 0, -1):
        # V_next = V[stage]

        # new cost matrix with subsequent stage costs
        new_cost = np.zeros(cost.shape)
        for state_num in len(dynamics):
            new_cost[state_num] = V[stage][state_num]


        # Calculate Vminmax and Vmaxmin
        Vminmax = np.min(np.max(cost + V[stage], axis=2), axis=1)
        Vmaxmin = np.max(np.min(cost + V[stage], axis=1), axis=2)

        # Check if saddle-point can be found
        if not np.array_equal(Vminmax, Vmaxmin):
            print("Must find mixed policy")

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
    print(states, "\n")

    control_inputs = generate_control_inputs()
    print(control_inputs, "\n")

    costs = generate_costs(states, control_inputs)
    print(costs, "\n")

    dynamics = generate_dynamics(states, control_inputs)
    print(dynamics, "\n")

    # ctg = generate_cost_to_go_ttt(2, costs, dynamics, states)
    # print(ctg, "\n")





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
