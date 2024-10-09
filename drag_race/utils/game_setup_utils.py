"""
Game Setup Utilities

State, Cost, and Dynamics functions
"""
import numpy as np
from drag_race.utils.upkeep_utils import array_find

def collision_check(state, control_input1, control_input2):
    """
    Compares player positions and returns whether they are in the same position or collided during the next maneuver
    :param state: matrix of distance, lane, and velocity for each player
    :param control_input1: array of 2 control input int values for lane and speed change
    :param control_input2: array of 2 control input int values for lane and speed change
    :return: boolean
    """

    # pos check
    if state[0][0] == state[0][1] and state[1][0] == state[1][1]:
        return True

    # next state check
    next_state = dynamics(state, control_input1, control_input2)
    if next_state[0][0] == next_state[0][1] and next_state[1][0] == next_state[1][1]:
        return True

    # vehicles not turning
    if control_input1[0] == 0 and control_input2[0] == 0:
        return False

    # maneuver check
    is_same_lane_change = (control_input1[0]) == (-control_input2[0])
    is_same_speed = (control_input1[1] + state[2][0]) == (control_input2[1] + state[2][1])
    is_same_distance = state[0][0] == state[0][1]
    if is_same_lane_change and is_same_speed and is_same_distance:
        return True

    return False


def safety_cost(state, control_input1, control_input2, penalty_lst):
    """
    Compute the cost matrix associated with the current game state for each player

    :param state: matrix of distance, lane, and velocity for each player
    :param control_input1: array of 2 control input int values for lane and speed change
    :param control_input2: array of 2 control input int values for lane and speed change
    :param penalty_lst: list of costs associated for maintaining speed, deceleration,
    acceleration,  turns, and collisions
    :return: cost matrix with for safety and rank objectives for each player
    """

    control_input_trans = np.array([control_input1.T, control_input2.T])
    control_input = control_input_trans.T

    # check for collisions
    if collision_check(state, control_input1, control_input2):
        return penalty_lst[-1], penalty_lst[-1]
    else:
        # calculate safety
        safety = [0, 0]
        for i in range(0, control_input.ndim):
            risk = 0
            # maintain speed
            if control_input[1][i] == 0:
                risk += penalty_lst[0]
            # decelerate
            elif control_input[1][i] < 0:
                risk += penalty_lst[1]
            # accelerate
            elif control_input[1][i] > 0:
                risk += penalty_lst[2]
            # turn
            if control_input[0][i] != 0:
                risk += penalty_lst[3]
            # tie
            next_state = dynamics(state, control_input1, control_input2)
            p1_dist = next_state[0][0]
            p2_dist = next_state[0][1]
            if p1_dist - p2_dist == 0:
                risk += penalty_lst[-2]

            safety[i] = risk

        return safety[0], safety[1]


def rank_cost(state, control_input1, control_input2, penalty_lst):
    """
    Compute the cost matrix associated with the current game state for each player

    :param state: matrix of distance, lane, and velocity for each player
    :param control_input1: matrix of lane change and acceleration for player 1
    :param control_input2: matrix of lane change and acceleration for player 2
    :param penalty_lst: list of costs associated for maintaining speed, deceleration,
    acceleration,  turns, and collisions
    :return: single rank cost for both players, min and max with respect to player 1 position
    """

    # check for collisions
    if collision_check(state, control_input1, control_input2):
        rank_cost = np.array([penalty_lst[-1], penalty_lst[-1]])
    else:
        # calculate rank
        next_state = dynamics(state, control_input1, control_input2)
        p1_dist = next_state[0][0]
        p2_dist = next_state[0][1]
        if p1_dist-p2_dist == 0:
            rank_cost = np.array([penalty_lst[-2], penalty_lst[-2]])
        else:
            rank_cost = np.array([p2_dist-p1_dist, p1_dist-p2_dist])

        # p1_lane = next_state[1][0]
        # p2_lane = next_state[1][1]

        # if p1_lane == p2_lane:
        #     rank_cost *= 2

    return rank_cost[0], rank_cost[1]


def check_state(state):
    """
    Verifies if state is possible
    :param state: numpy array with state info
    :return: bool if state is possible
    """
    # invalid starting line conditions
    # position (wrong lane)
    if state[0][0] == 0 and state[1][0] == 1:
        return False
    elif state[0][1] == 0 and state[1][1] == 0:
        return False
    # velocity (initial speed)
    if state[0][0] == 0 and state[2][0] == 1:
        return False
    elif state[0][1] == 0 and state[2][1] == 1:
        return False
    else:
        return True


def generate_states(k):
    """
    Calculate all possible states over k rounds
    :param k: the number of stages in game
    :return: list of all possible numpy array states
    """
    # generate all possible states for a player
    player_state_lst = []
    for distance in range(0, k+2):
        for lane in range(0, 2):
            for velocity in range(0, 2):
                state = [distance, lane, velocity]
                player_state_lst.append(state)

    # make list of all combinations for both players
    state_lst = []
    cnt = 0
    for player_state1 in player_state_lst:
        for player_state2 in player_state_lst:
            state = np.array([player_state1, player_state2]).T

            if check_state(state):
                state_lst.append(state)
            # else:
            #     cnt +=1
            #     print(state,cnt,"\n")
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

            control_input_lst.append(np.array([i, j]))

    return control_input_lst


def generate_costs(state_lst, control_input_lst, penalty_lst, cost_function):
    """
    Calculate stage cost given each state and control input
    :param state_lst: list of state arrays
    :param control_input_lst: list of all possible control input arrays of lane change and acceleration values
    :param penalty_lst: list of costs associated for maintaining speed, deceleration, acceleration,  turns, collisions
    :param cost_function: function used as objective (safety or rank)
    :return: tensor of stage cost indexed by each state/control input
    """

    cost1 = np.empty((len(state_lst), len(control_input_lst), len(control_input_lst)), dtype=float) * np.nan
    cost2 = cost1.copy()

    # for k in range(stage_count):
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            for l in range(len(control_input_lst)):

                next_state_mat = dynamics(state_lst[i], control_input_lst[j], control_input_lst[l])
                next_state_index = array_find(next_state_mat, state_lst)
                if next_state_index != -1:
                    cost1[i, j, l], cost2[i, j, l] = cost_function(state_lst[i], control_input_lst[j],
                                                                   control_input_lst[l], penalty_lst)

    return cost1, cost2


def dynamics(state, control_input1, control_input2):
    """
    Calculate next state from given current state and control input
    :param state: matrix of distance, lane, and velocity for each player
    :param control_input1: player one lane and velocity change matrix
    :param control_input2: player two lane and velocity change matrix
    :return: next state matrix
    """
    control_input_trans = np.array([control_input1.T, control_input2.T])
    control_input = control_input_trans.T

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
    Make lookup table for next state given current state and player actions
    :param state_lst: list of all possible state arrays
    :param control_input_lst: list of all possible control input arrays
    :return: array of dynamics, state x control_input1 x control_input2
    """
    dynamics_lookup_mat = np.empty((len(state_lst), len(control_input_lst),
                                    len(control_input_lst)), dtype=int) * np.nan
    # for k in range(stage_count):
    for i in range(len(state_lst)):
        for j in range(len(control_input_lst)):
            for l in range(len(control_input_lst)):
                next_state_mat = dynamics(state_lst[i], control_input_lst[j], control_input_lst[l])
                next_state_index = array_find(next_state_mat, state_lst)
                if next_state_index != -1:
                    dynamics_lookup_mat[i, j, l] = next_state_index

    return dynamics_lookup_mat