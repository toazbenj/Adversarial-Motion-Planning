import numpy as np
from drag_race.utilities import generate_states, mixed_policy_3d

stage_count = 1
state_lst = generate_states(stage_count)

cost1 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan,   np.nan, np.nan,   np.nan,   np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan,  1., np.nan, 4.,   np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan,  3, np.nan, -1.,  np.nan, np.nan, np.nan, np.nan]])
cost2 = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan,   np.nan, np.nan,   np.nan,   np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan,  np.nan, np.nan, np.nan,   np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan, 0, np.nan, np.nan, 1, np.nan, np.nan, np.nan],
       [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
       [np.nan, np.nan,  -1, np.nan, np.nan,  0, np.nan, np.nan, np.nan]])

big_cost = np.zeros((len(state_lst), 9, 9))
for i in range(len(state_lst)):
    if i < 40:
        big_cost[i] = cost1
    else:
           big_cost[i] = cost2

row_policy, col_policy, ctg = mixed_policy_3d(big_cost, state_lst, stage_count)
print(row_policy)
print(ctg)