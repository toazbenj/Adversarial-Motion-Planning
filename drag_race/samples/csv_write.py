import numpy as np
import csv

# Given variables and functions (you should replace these functions with the actual implementations)
stage_count = 1
penalty_lst = [0, 1, 2, 1, 1]
init_state = np.array([[0, 0],
                       [0, 1],
                       [0, 0]])


def generate_states(stage_count):
    # Dummy implementation (replace with actual function)
    return np.random.rand(5, 2)  # Example state array


def generate_control_inputs():
    # Dummy implementation (replace with actual function)
    return np.random.rand(3, 2)  # Example control inputs array


def generate_costs(states, control_inputs, penalty_lst, rank_cost):
    # Dummy implementation (replace with actual function)
    return np.random.rand(5, 3), np.random.rand(5, 3)  # Example costs arrays


def generate_dynamics(states, control_inputs):
    # Dummy implementation (replace with actual function)
    return np.random.rand(5, 2, 2)  # Example dynamics array


def generate_cost_to_go_mixed(stage_count, costs1, costs2, control_inputs, states):
    # Dummy implementation (replace with actual function)
    return np.random.rand(5, 2), np.random.rand(5, 2), np.random.rand(5, 2), np.random.rand(5,
                                                                                            2)  # Example ctg and policy arrays

def write_variables_to_csv(filename, variables):
    """
    Write variables to a csv file
    :param filename: name of data file str
    :param variables: tuple of offline calculated game variables
    """
    keys = ['stage_count', 'penalty_lst', 'init_state', 'states', 'control_inputs',
            'costs1', 'costs2', 'dynamics', 'ctg1', 'ctg2', 'policy1', 'policy2']

    # Flatten numpy arrays and convert to lists for writing to CSV
    flat_variables = {}
    for key, value in zip(keys, variables):
        if isinstance(value, np.ndarray):
            flat_variables[key] = value.flatten().tolist()
        else:
            flat_variables[key] = value

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['key', 'values'])

        # Write data
        for key, value in flat_variables.items():
            if isinstance(value, list):
                csvwriter.writerow([key] + value)
            else:
                csvwriter.writerow([key, value])

# Generate the required data
states = generate_states(stage_count)
control_inputs = generate_control_inputs()
costs1, costs2 = generate_costs(states, control_inputs, penalty_lst, rank_cost=None)
dynamics = generate_dynamics(states, control_inputs)
ctg1, ctg2, policy1, policy2 = generate_cost_to_go_mixed(stage_count, costs1, costs2, control_inputs, states)

variables = (stage_count, penalty_lst, init_state, states, control_inputs, costs1, costs2,
                           dynamics, ctg1, ctg2, policy1, policy2)
write_variables_to_csv('output.csv', variables)
