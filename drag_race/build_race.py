"""
Build Race with Cost to Go and Vector Payoffs
Created by Ben Toaz on 6-7-24

2 player drag race with different payoffs for each player, security policies for safety and rank bimatrix games with
both players as minimizers. Builds cost structure and dynamics of game, calculates optimal policies, saves them in an
npz file for loading into game play program.

Run this first, generate the saved_game.npz file with all game structure data and policies. Then run play_race.py to
get results.
"""

import numpy as np
from utils.game_setup_utils import (generate_costs, generate_states, generate_dynamics,
                                    generate_control_inputs, rank_cost, safety_cost)
from utils.policy_utils import (bimatrix_mixed_policy, generate_moderate_policies, mixed_policy_3d, security_value,
                                clean_matrix, pure_policy_mapping)
from utils.upkeep_utils import expand_mat, write_npz_build, remap_matrix
from utils.cost_adjust_utils import cost_adjustment, add_errors

def generate_cost_to_go_mixed(stage_count, costs1, costs2, control_inputs, state_lst):
    """
    Calculates cost to go for each state at each stage (approximate mixed nash equilibrium)
    :param stage_count: decision point count int
    :param costs1: cost array of state x control input x control input
    :param costs2: cost array of state x control input x control input
    :param control_inputs: list of control input arrays
    :param state_lst: list of all possible state arrays
    :return: array of policies for each player of state x control input
    """
    # Initialize cost to go with zeros
    V1 = np.zeros((stage_count + 2, len(costs1)))
    V2 = np.zeros((stage_count + 2, len(costs1)))
    policy1 = np.zeros((stage_count + 1, len(state_lst), len(control_inputs)))
    policy2 = np.zeros((stage_count + 1, len(state_lst), len(control_inputs)))

    # Iterate backwards from k to 1
    for stage in range(stage_count, -1, -1):
        combined_cost1 = expand_mat(V1[stage + 1], costs1) + costs1
        combined_cost2 = expand_mat(V2[stage + 1], costs2) + costs2

        policy1[stage], policy2[stage], mixed_value1, mixed_value2 = bimatrix_mixed_policy(combined_cost1,
                                                                                           combined_cost2,
                                                                                           state_lst, stage_count)
        V1[stage] = mixed_value1
        V2[stage] = mixed_value2

    return policy1, policy2


def generate_cost_to_go_security(stage_count, costs1, costs2, control_inputs, state_lst):
    """
    Calculates cost to go for each state at each stage (security policies)
    :param stage_count: decision point count int
    :param costs1: cost array of state x control input x control input
    :param costs2: cost array of state x control input x control input
    :param control_inputs: list of control input arrays
    :param state_lst: list of all possible state arrays
    :return: array of policies for each player of state x control input
    """
    # Initialize cost to go with zeros
    V1 = np.zeros((stage_count + 2, len(costs1)))
    V2 = np.zeros((stage_count + 2, len(costs1)))
    policy1 = np.zeros((stage_count + 1, len(state_lst), len(control_inputs)))
    policy2 = np.zeros((stage_count + 1, len(state_lst), len(control_inputs)))

    # Iterate backwards from k to 1
    for stage in range(stage_count, -1, -1):
        combined_cost1 = expand_mat(V1[stage + 1], costs1) + costs1
        combined_cost2 = expand_mat(V2[stage + 1], costs2) + costs2

        policy1[stage], _, mixed_value1 = mixed_policy_3d(combined_cost1, state_lst, stage_count)
        _, policy2[stage], mixed_value2 = mixed_policy_3d(combined_cost2, state_lst, stage_count, is_min_max=False)

        V1[stage] = mixed_value1
        V2[stage] = mixed_value2

    return policy1, policy2


def generate_cost_to_go_adjusted(stage_count, rank_cost1, rank_cost2, safety_cost1, safety_cost2, control_inputs,
                                 states, init_state, dynamics):
    """
    Calculates cost to go for each state at each stage (security policies)
    :param stage_count: decision point count int
    :param costs1: cost array of state x control input x control input
    :param costs2: cost array of state x control input x control input
    :param control_inputs: list of control input arrays
    :param state_lst: list of all possible state arrays
    :return: array of policies for each player of state x control input
    """
    # adjust cost
    player1_costs = [safety_cost1, rank_cost1]
    player2_costs = [safety_cost2, rank_cost2]

    num_states = safety_cost1.shape[0]
    player1_errors = [np.zeros(safety_cost1.shape),
                      np.zeros(safety_cost1.shape)]

    for i in range(len(player1_costs)):
        for state in range(num_states):
            state_cost1 = player1_costs[i][state]
            state_cost2 = player2_costs[i][state]
            state_error = cost_adjustment(clean_matrix(state_cost1), clean_matrix(state_cost2))

            player1_errors[i][state] = remap_matrix(state_cost1, state_error)

    player1_adjusted_costs = add_errors(player1_errors, player1_costs)

    # p1 plays pure sec policy of adjusted rank costs, p2 plays pure sec policy of original safety costs
    costs1 = player1_adjusted_costs[1]
    costs2 = rank_cost2

    # cost to go calculation
    V1 = np.zeros((stage_count + 2, len(costs1)))
    V2 = np.zeros((stage_count + 2, len(costs1)))

    # Iterate backwards from k to 1
    for stage in range(stage_count, -1, -1):
        combined_cost1 = expand_mat(V1[stage + 1], costs1) + costs1
        combined_cost2 = expand_mat(V2[stage + 1], costs2) + costs2
        V1[stage], V2[stage] = security_value(combined_cost1, combined_cost2, states, stage_count)

    control1, control2, _ = optimal_actions(stage_count, costs1, costs2, V1, V2, dynamics, init_state)

    policy1 = pure_policy_mapping(control1, states, control_inputs, stage_count)
    policy2 = pure_policy_mapping(control2, states, control_inputs, stage_count)

    return policy1, policy2


def optimal_actions(stage_count, costs1, costs2, V1, V2, dynamics, init_state):
    """
    Given initial state, play actual game, calculate best control inputs and tabulate state at each stage
    :param k: stage count
    :param cost: cost array of state x control input 1 x control input 2
    :param ctg: cost to go array of stage x state
    :param dynamics: next state array given control inputs of state x control input 1 x control input 2
    :param initial_state: index of current state int
    :return: list of best control input indicies for each player, states played in the game
    """
    control1 = np.zeros(stage_count+1, dtype=int)
    control2 = np.zeros(stage_count+1, dtype=int)
    states_played = np.zeros(stage_count+2, dtype=int)
    # states_played[0] = init_state

    for stage in range(stage_count+1):
        V_last1 = V1[stage+1]
        shape_tup = costs1.shape
        repeat_int = np.prod(shape_tup)//np.prod(V_last1.shape)
        V_expanded1 = np.repeat(V_last1[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        V_last2 = V2[stage + 1]
        shape_tup = costs1.shape
        repeat_int = np.prod(shape_tup) // np.prod(V_last2.shape)
        V_expanded2 = np.repeat(V_last2[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)

        control1[stage] = np.nanargmin(np.nanmax(costs1[states_played[stage]] + V_expanded1[states_played[stage]], axis=1),
                                       axis=0)
        control2[stage] = np.nanargmax(np.nanmin(costs2[states_played[stage]] + V_expanded2[states_played[stage]], axis=0),
                                       axis=0)

        states_played[stage + 1] = dynamics[states_played[stage], control1[stage], control2[stage]]

    return control1, control2, states_played


def build_race(model_path, stage_count, type, rank_penalty_lst, safety_penalty_lst, init_state):
    """
    Populates all game variables for drag race
    :param model_path: str path to save game variables
    :param stage_count: int number of decision epochs
    :param type: string of game policy type
    :param rank_penalty_lst: list of floats, action costs: maintain speed, decelerate, accelerate, turn, tie, collide
    :param safety_penalty_lst: list of floats, action costs: maintain speed, decelerate, accelerate, turn, tie, collide
    :param init_state: array of game state, player x property
    """
    states = generate_states(stage_count)
    control_inputs = generate_control_inputs()

    rank_cost1, rank_cost2 = generate_costs(states, control_inputs, rank_penalty_lst, rank_cost)
    safety_cost1, safety_cost2 = generate_costs(states, control_inputs, safety_penalty_lst, safety_cost)

    dynamics = generate_dynamics(states, control_inputs)

    match type:
        case 'mixed_equilibrium':
            aggressive_policy1, aggressive_policy2 = generate_cost_to_go_mixed(stage_count, rank_cost1, rank_cost2,
                                                                               control_inputs, states)
            conservative_policy1, conservative_policy2 = generate_cost_to_go_mixed(stage_count, safety_cost1,
                                                                                   safety_cost2, control_inputs, states)

        case 'security_policies':
            aggressive_policy1, aggressive_policy2 = generate_cost_to_go_security(stage_count, rank_cost1, rank_cost2,
                                                                                  control_inputs, states)
            conservative_policy1, conservative_policy2 = generate_cost_to_go_security(stage_count, safety_cost1,
                                                                                      safety_cost2, control_inputs,
                                                                                      states)
        case 'adjusted_costs':
            aggressive_policy1, aggressive_policy2 = generate_cost_to_go_adjusted(stage_count, rank_cost1, rank_cost2,
                                                                                  safety_cost1, safety_cost2,
                                                                                  control_inputs, states, init_state,
                                                                                  dynamics)

            conservative_policy1, conservative_policy2 = aggressive_policy1, aggressive_policy2
        case _:
            print("Unrecognized type")


    moderate_policy1, moderate_policy2 = generate_moderate_policies(aggressive_policy1, aggressive_policy2,
                                                                    conservative_policy1, conservative_policy2)

    write_npz_build(model_path, (stage_count, rank_penalty_lst, safety_penalty_lst, init_state, states,
                    control_inputs, rank_cost1, rank_cost2, safety_cost1, safety_cost2, dynamics,
                    aggressive_policy1, aggressive_policy2, conservative_policy1, conservative_policy2,
                    moderate_policy1, moderate_policy2))


if __name__ == '__main__':
    type = 'adjusted_costs'
    # type = 'mixed_equilibrium'
    # type = 'security_policies'

    model_path = "offline_calcs/"+type+"_build.npz"
    stage_count = 2
    # maintain speed, decelerate, accelerate, turn, tie, collide
    rank_penalty_lst = [2, 3, 0, 1, 5, 10]
    safety_penalty_lst = [0, 1, 3, 2, 1, 10]
    init_state = np.array([[0, 0],
                           [0, 1],
                           [0, 0]])

    build_race(model_path, stage_count, type, rank_penalty_lst, safety_penalty_lst, init_state)
