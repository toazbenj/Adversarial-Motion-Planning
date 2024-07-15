"""
Generate Race Visualizations

Run third after play_race.py. Clean out the plot directory and generate images for each race, graphs of costs.
"""
from utilities import read_npz_play
from graphics import plot_race, plot_pareto_front, plot_average_cost
import os
import glob

result_filename = "results_test.npz"
average_game_values, states_played, states, pair_labels = read_npz_play(result_filename)

# clear existing plots
files = glob.glob("plots/Race_Visuals/Aggressive-Aggressive/*") + \
    glob.glob("plots/Race_Visuals/Conservative-Aggressive/*") + \
    glob.glob("plots/Race_Visuals/Conservative-Conservative/*") + \
    glob.glob("plots/Race_Visuals/Moderate-Aggressive/*") + \
    glob.glob("plots/Race_Visuals/Moderate-Conservative/*") + \
    glob.glob("plots/Race_Visuals/Moderate-Moderate/*") + \
    glob.glob("plots/Pareto_Fronts/*") + glob.glob("plots/Average_Costs/*")

for f in files:
    os.remove(f)

for i in range(len(pair_labels)):
    plot_average_cost(average_game_values[i], pair_labels[i])
    #
    # for j in range(50):
    #     plot_race(states_played[i, j], states, pair_labels[i], j)

plot_pareto_front(average_game_values, pair_labels)
