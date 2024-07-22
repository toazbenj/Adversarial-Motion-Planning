"""
Generate Race Visualizations

Run third after play_race.py. Clean out the plot directory and generate images for each race, graphs of costs.
WARNING: Will wipe all existing plots!
"""
from utilities import read_npz_play
from graphics import plot_race_view, plot_pareto_front, plot_average_cost
import os
import glob

def plot_race(play_path, plot_directory, sample_races):
    average_game_values, states_played, states, pair_labels = read_npz_play(play_path)

    # clear existing plots
    files = glob.glob("plots/" + plot_directory + "/Race_Visuals/Aggressive-Aggressive/*") + \
            glob.glob("plots/" + plot_directory + "/Race_Visuals/Conservative-Aggressive/*") + \
            glob.glob("plots/" + plot_directory + "/Race_Visuals/Conservative-Conservative/*") + \
            glob.glob("plots/" + plot_directory + "/Race_Visuals/Moderate-Aggressive/*") + \
            glob.glob("plots/" + plot_directory + "/Race_Visuals/Moderate-Conservative/*") + \
            glob.glob("plots/" + plot_directory + "/Race_Visuals/Moderate-Moderate/*") + \
            glob.glob("plots/" + plot_directory + "/Pareto_Fronts/*") + glob.glob(
        "plots/" + plot_directory + "/Average_Costs/*")

    for f in files:
        os.remove(f)

    # number of times probabilistic policies are played out as actual choices for example plots
    for i in range(len(pair_labels)):
        plot_average_cost(average_game_values[i], pair_labels[i], plot_directory)

        for j in range(sample_races):
            plot_race_view(states_played[i, j], states, pair_labels[i], plot_directory, j)

    plot_pareto_front(average_game_values, pair_labels, plot_directory)

if __name__ == '__main__':
    play_path = "offline_calcs/security_play.npz"
    plot_directory = "security_policies"
    # result_filename = "offline_calcs/mixed_play.npz"
    # plot_directory = "mixed_equilibrium"
    sample_races = 0

    plot_race(play_path, plot_directory, sample_races)
