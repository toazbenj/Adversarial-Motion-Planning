"""
Upkeep Utilities

Matrix manipulation and data saving functions.
"""
import numpy as np

def expand_mat(small_mat, big_mat):
    """
    Helper function for making small matrix match dimensions of large one by repeating values
    :param small_mat: numpy array to expand
    :param big_mat: numpy array to match
    :return: expanded numpy array
    """
    shape_tup = big_mat.shape
    repeat_int = np.prod(shape_tup) // np.prod(small_mat.shape)
    expanded_mat = np.repeat(small_mat[:, np.newaxis, np.newaxis], repeat_int).reshape(shape_tup)
    return expanded_mat


def clean_matrix(mat):
    """
    Remove rows/cols with all NaNs, keep matrix shape
    :param mat: array with NaN values
    :return: numpy array with no nans, retains relative position of real values
    """
    mat = mat[~np.isnan(mat).all(axis=1)]
    mat = mat[:, ~np.isnan(mat).all(axis=0)]
    return mat


def array_find(value_array, search_lst):
    """
    Find the index of the array within a list of arrays
    :param value_array: array to search for
    :param search_lst: list to search within
    :return: index of array, -1 if not found
    """
    for idx, item in enumerate(search_lst):
        if np.array_equal(value_array, item):
            return idx
    return -1


def remap_values(mat, small_arr, is_row=True):
    """
    Map values from small arr to large arr, each large arr index corresponds to non nan value
    in either row or col of mat
    :param mat: mat of cost values
    :param small_arr: list of probabilities from cleaned payoff matrix
    :param is_row: policies for row or col
    :return: arr with non nan value indexes with probabilities, rest set to 0
    """
    large_arr = np.zeros(mat.shape[1])
    small_lst = list(small_arr)

    if is_row:
        mapping_arr = np.isfinite(mat[:, ~np.isnan(mat).all(axis=0)])[:, 0]
    else:
        mapping_arr = np.isfinite(mat[~np.isnan(mat).all(axis=1)])[0]

    for i in range(len(large_arr)):
        if mapping_arr[i]:
            large_arr[i] = small_lst.pop(0)
        i += 1
    return large_arr


def write_npz_build(filename, variables):
    """
    Write variables to a npz file
    :param filename: name of data file str
    :param variables: tuple of offline calculated game variables
    """
    keys = ["stage_count", "rank_penalty_lst", "safety_penalty_lst", "init_state", "states", "control_inputs",
            "rank_cost1", "rank_cost2", "safety_cost1", "safety_cost2", "dynamics",
            "aggressive_policy1", "aggressive_policy2", "conservative_policy1", "conservative_policy2",
            "moderate_policy1", "moderate_policy2"]

    data_dict = {key: value for key, value in zip(keys, variables)}
    np.savez(filename, **data_dict)


def read_npz_build(filename):
    """
    Load variables from npz file
    :param filename: name of data file str
    :return: tuple of offline calculated game variables
    """
    with np.load(filename, allow_pickle=True) as data:

        stage_count = data['stage_count'].item()
        rank_penalty_lst = data['rank_penalty_lst'].tolist()
        safety_penalty_lst = data['safety_penalty_lst'].tolist()

        init_state = data['init_state']
        states = data['states']
        control_inputs = data['control_inputs']

        rank_cost1 = data['rank_cost1']
        rank_cost2 = data['rank_cost2']
        safety_cost1 = data['safety_cost1']
        safety_cost2 = data['safety_cost2']

        dynamics = data['dynamics']

        aggressive_policy1 = data['aggressive_policy1']
        aggressive_policy2 = data['aggressive_policy2']
        conservative_policy1 = data['conservative_policy1']
        conservative_policy2 = data['conservative_policy2']
        moderate_policy1 = data['moderate_policy1']
        moderate_policy2 = data['moderate_policy2']

    return stage_count, rank_penalty_lst, safety_penalty_lst, init_state, states, control_inputs, \
        rank_cost1, rank_cost2, safety_cost1, safety_cost2, dynamics, \
        aggressive_policy1, aggressive_policy2, conservative_policy1, conservative_policy2, \
        moderate_policy1, moderate_policy2


def write_npz_play(filename, variables):
    """
    Write variables to a npz file
    :param filename: name of data file str
    :param variables: tuple of offline calculated game variables
    """
    keys = ["average_game_values", "states_played", "states", "pair_labels"]
    data_dict = {key: value for key, value in zip(keys, variables)}
    np.savez(filename, **data_dict)


def read_npz_play(filename):
    """
    Load variables from npz file
    :param filename: name of data file str
    :return: tuple of offline calculated game variables
    """
    with np.load(filename, allow_pickle=True) as data:
        average_game_values = data["average_game_values"]
        states_played = data["states_played"]
        states = data['states']
        pair_labels = data["pair_labels"]

    return average_game_values, states_played, states, pair_labels


import numpy as np


def remap_matrix(example_mat, small_mat):
    """
    Map values from small matrix into larger example matrix. Real-valued entries in example_mat
    are replaced by corresponding values from small_mat, while other entries are set to NaN.

    :param example_mat: Large matrix with some entries as real values and others as NaN
    :param small_mat: Smaller matrix with values to fill into real-valued positions of example_mat
    :return: Matrix with real-valued positions from example_mat filled with values from small_mat, others set to NaN
    """
    # Ensure example_mat is flattened
    flat_example = example_mat.flatten()
    # Create an output matrix of the same shape, defaulting to NaN
    output_mat = np.full_like(flat_example, np.nan, dtype=np.float64)

    # Extract positions of finite (real) values in example_mat
    real_positions = np.isfinite(flat_example)

    # Check if the number of real positions matches the length of small_mat
    if int(real_positions.sum()) != len(small_mat.flatten()):
        raise ValueError("Number of non-NaN values in example_mat does not match the size of small_mat")

    # Fill real positions in the output matrix with values from small_mat
    output_mat[real_positions] = small_mat.flatten()

    # Reshape to the original dimensions of example_mat
    return output_mat.reshape(example_mat.shape)
