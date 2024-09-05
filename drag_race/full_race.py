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
is_mixed_equilibrium = False
if is_mixed_equilibrium:
    build_path = "offline_calcs/mixed_build.npz"
    play_path = "offline_calcs/mixed_play.npz"
    plot_directory = "mixed_equilibrium"
else:
    build_path = "offline_calcs/security_build.npz"
    play_path = "offline_calcs/security_play.npz"
    plot_directory = "security_policies"

stage_count = 3
# maintain speed, decelerate, accelerate, turn, tie, collide
rank_penalty_lst = [0, 1, 2, 1, 5, 10]
safety_penalty_lst = [0, 3, 3, 3, 5, 20]
init_state = np.array([[0, 0],
                       [0, 1],
                       [0, 0]])

# Play Parameters =====================================================================================================
is_verbose = True
race_count = 100

# Plot Parameters =====================================================================================================
sample_races = 50

# Calculations ========================================================================================================
build_race(build_path, stage_count, is_mixed_equilibrium, rank_penalty_lst, safety_penalty_lst, init_state)
play_race(build_path, play_path, is_verbose, race_count)
plot_race(play_path, plot_directory, sample_races)
