import numpy as np

def write_variables_to_npz(filename, variables):
    """
    Write variables to a npz file
    :param filename: name of data file str
    :param variables: tuple of offline calculated game variables
    """
    keys = ['stage_count', 'penalty_lst', 'init_state', 'states', 'control_inputs',
            'costs1', 'costs2', 'dynamics', 'ctg1', 'ctg2', 'policy1', 'policy2']

    data_dict = {key: value for key, value in zip(keys, variables)}
    np.savez(filename, **data_dict)


import numpy as np

def read_npz_to_variables(filename):
    """
    Load variables from npz file
    :param filename: name of data file str
    :return: tuple of offline calculated game variables
    """
    data = np.load(filename, allow_pickle=True)

    stage_count = data['stage_count'].item()
    penalty_lst = data['penalty_lst'].tolist()

    init_state = data['init_state']
    states = data['states']
    control_inputs = data['control_inputs']

    costs1 = data['costs1']
    costs2 = data['costs2']

    dynamics = data['dynamics']

    ctg1 = data['ctg1']
    ctg2 = data['ctg2']
    policy1 = data['policy1']
    policy2 = data['policy2']

    return (stage_count, penalty_lst, init_state, states, control_inputs, costs1, costs2, dynamics, ctg1, ctg2, policy1, policy2)




# Example usage
variables = (
    5,  # stage_count
    [1, 2, 3],  # penalty_lst
    np.array([[0, 0], [1, 1], [2, 2]]),  # init_state
    np.array([[[0, 0], [0, 1], [0, 0]], [[0, 1], [0, 0], [0, 0]]]),  # states
    np.array([[0, 1], [1, 0]]),  # control_inputs
    np.array([[1, 2, 3], [4, 5, 6]]),  # costs1
    np.array([[7, 8, 9], [10, 11, 12]]),  # costs2
    np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # dynamics
    np.array([[1, 2], [3, 4]]),  # ctg1
    np.array([[5, 6], [7, 8]]),  # ctg2
    np.array([[9, 10], [11, 12]]),  # policy1
    np.array([[13, 14], [15, 16]])  # policy2
)

write_variables_to_npz('variables.npz', variables)


# Example usage
filename = 'variables.npz'
variables = read_npz_to_variables(filename)
print(variables)
