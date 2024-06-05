import numpy as np


def collision_check(state, control_input):
    """
    Compares player positions and returns whether they are in the same position or collided during the next maneuver
    :param state: matrix of distance, lane, and velocity for each player
    :param control_input: matrix of lane change and acceleration for each player
    :return: boolean
    """

    for i in range(state.ndim):
        for j in range(state.ndim):
            if i == j:
                continue

            # pos check
            if state[0][i] == state[0][j] and state[1][i] == state[1][j]:
                return True

            # maneuver check
            # vehicles not turning
            if np.array_equal(control_input[0], [0, 0]):
                return False

            is_same_lane_change = control_input[0][i] == -control_input[0][j]
            is_same_speed = control_input[1][i] + state[2][i] == control_input[1][i] + state[2][j]
            is_same_distance = state[0][i] == state[0][j]
            if is_same_lane_change and is_same_speed and is_same_distance:
                return True
    return False


def cost(state, control_input, penalty_lst):
    """
    Compute the cost matrix associated with the current game state for each player

    :param state: matrix of distance, lane, and velocity for each player
    :param control_input: matrix of lane change and acceleration for each player
    :param penalty_lst: list of costs associated for maintaining speed, deceleration,
    acceleration,  turns, and collisions
    :return: cost matrix with for safety and rank objectives for each player
    """
    # check for collisions
    collision_penalty_int = penalty_lst[-1]
    if collision_check(state, control_input):
        return np.array([[collision_penalty_int, collision_penalty_int],
                        [collision_penalty_int, collision_penalty_int]])
    else:
        # calculate rank
        ranks = [1, 1]
        p1_dist = state[0][0] + control_input[1][0]
        p2_dist = state[0][1] + control_input[1][1]
        if p1_dist > p2_dist:
            ranks = [0, 1]
        elif p1_dist < p2_dist:
            ranks = [1, 0]

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

            safety[i] = risk

        return np.array([safety,
                        ranks])


def cost_to_go(state, control_input, penalty_lst):
    """
    Compute the cost to go for each round
    :param state:
    :param control_input:
    :param penalty_lst:
    :return:
    """

if __name__ == "__main__":
    state = np.array([[0, 1],
                      [0, 1],
                      [0, 0]])
    control_input = np.array([[0, -1],
                              [1, 1]])
    # maintaining speed, deceleration, acceleration,  turns, and collisions
    penalty_lst = [0, 1, 2, 1, 10]
    cost = cost(state, control_input, penalty_lst)
    print(cost)