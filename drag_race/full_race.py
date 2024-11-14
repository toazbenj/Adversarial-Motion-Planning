"""
Full Race

Builds, plays, and plots drag race all in one go. All parameters for each subprogram are available to be adjusted in one
place. You're welcome.
"""

import numpy as np
from build_race import build_race
from play_race import play_race
from plot_race import plot_race

# Build Parameters ====================================================================================================
# type = 'adjusted_costs'
type = 'mixed_equilibrium'
# type = 'security_policies'

model_path = "offline_calcs/"+type+"_build.npz"
build_path = "offline_calcs/"+type+"_build.npz"
play_path = "offline_calcs/"+type+"_play.npz"
plot_directory = type

stage_count = 2
# maintain speed, decelerate, accelerate, turn, tie, collide
rank_penalty_lst = [0, 1, 2, 1, 5, 10]
safety_penalty_lst = [0, 1, 3, 3, 3, 10]
init_state = np.array([[0, 0],
                       [0, 1],
                       [0, 0]])

# Play Parameters =====================================================================================================
is_verbose = True
race_count = 1000

# Plot Parameters =====================================================================================================
sample_races = 10

# Calculations ========================================================================================================
build_race(build_path, stage_count, type, rank_penalty_lst, safety_penalty_lst, init_state)
play_race(build_path, play_path, is_verbose, race_count)
plot_race(play_path, plot_directory, sample_races)
