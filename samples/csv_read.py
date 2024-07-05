import numpy as np
import csv
import ast  # Import ast for safely parsing string representations of lists


def read_csv_to_variables(filename):
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip header
        next(csvreader)

        data_dict = {}
        for row in csvreader:
            key = row[0]
            values = row[1:]
            if len(values) == 1:
                # Convert single value to int or float
                value = values[0]
                if value.replace('.', '', 1).isdigit():
                    data_dict[key] = float(value) if '.' in value else int(value)
                else:
                    # Safely parse string representation of lists using ast.literal_eval
                    data_dict[key] = ast.literal_eval(value)
            else:
                # Handle lists of values
                data_dict[key] = [float(v) if '.' in v else int(v) for v in values]

    # Assign variables from the dictionary
    stage_count = data_dict['stage_count']
    penalty_lst = data_dict['penalty_lst']

    init_state = np.array(data_dict['init_state']).reshape((3, 2))
    states = np.array(data_dict['states']).reshape((-1, 2))  # Adjust shape as per your actual states' shape
    control_inputs = np.array(data_dict['control_inputs']).reshape(
        (-1, 2))  # Adjust shape as per your actual control inputs' shape

    costs1 = np.array(data_dict['costs1']).reshape((-1, 3))  # Adjust shape as per your actual costs1 shape
    costs2 = np.array(data_dict['costs2']).reshape((-1, 3))  # Adjust shape as per your actual costs2 shape

    dynamics = np.array(data_dict['dynamics']).reshape((-1, 2, 2))  # Adjust shape as per your actual dynamics shape

    ctg1 = np.array(data_dict['ctg1']).reshape((-1, 2))  # Adjust shape as per your actual ctg1 shape
    ctg2 = np.array(data_dict['ctg2']).reshape((-1, 2))  # Adjust shape as per your actual ctg2 shape
    policy1 = np.array(data_dict['policy1']).reshape((-1, 2))  # Adjust shape as per your actual policy1 shape
    policy2 = np.array(data_dict['policy2']).reshape((-1, 2))  # Adjust shape as per your actual policy2 shape

    return (stage_count, penalty_lst, init_state, states, control_inputs, costs1, costs2, dynamics, ctg1, ctg2, policy1,
            policy2)


# Call the function and retrieve the variables
(stage_count, penalty_lst, init_state, states, control_inputs, costs1, costs2, dynamics, ctg1, ctg2, policy1,
 policy2) = read_csv_to_variables('output_data.csv')

# Print to verify the data is read correctly
print("stage_count:", stage_count)
print("penalty_lst:", penalty_lst)
print("init_state:\n", init_state)
print("states:\n", states)
print("control_inputs:\n", control_inputs)
print("costs1:\n", costs1)
print("costs2:\n", costs2)
print("dynamics:\n", dynamics)
print("ctg1:\n", ctg1)
print("ctg2:\n", ctg2)
print("policy1:\n", policy1)
print("policy2:\n", policy2)
