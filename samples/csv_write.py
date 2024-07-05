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


# Generate the required data
states = generate_states(stage_count)
control_inputs = generate_control_inputs()
costs1, costs2 = generate_costs(states, control_inputs, penalty_lst, rank_cost=None)
dynamics = generate_dynamics(states, control_inputs)
ctg1, ctg2, policy1, policy2 = generate_cost_to_go_mixed(stage_count, costs1, costs2, control_inputs, states)

# Open a CSV file to write the data
with open('output_data.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Variable', 'Value'])
    csvwriter.writerow(['stage_count', stage_count])
    csvwriter.writerow(['penalty_lst', penalty_lst])
    csvwriter.writerow(['init_state'] + init_state.flatten().tolist())
    csvwriter.writerow(['states'] + states.flatten().tolist())
    csvwriter.writerow(['control_inputs'] + control_inputs.flatten().tolist())
    csvwriter.writerow(['costs1'] + costs1.flatten().tolist())
    csvwriter.writerow(['costs2'] + costs2.flatten().tolist())
    csvwriter.writerow(['dynamics'] + dynamics.flatten().tolist())
    csvwriter.writerow(['ctg1'] + ctg1.flatten().tolist())
    csvwriter.writerow(['ctg2'] + ctg2.flatten().tolist())
    csvwriter.writerow(['policy1'] + policy1.flatten().tolist())
    csvwriter.writerow(['policy2'] + policy2.flatten().tolist())
    print("Data written")

print("Data successfully written to output_data.csv")
