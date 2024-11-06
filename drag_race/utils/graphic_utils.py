"""
Graphics

Functions to graph state output and values charts for drag race game
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_race_view(states_played, states, label, result_directory, number):
    """
    Visualize the race game by plotting the positions (distance and lane) of the players over time
    and their velocities on a separate graph, with each data point labeled with a timestamp.

    :param states_played: List of indices of states played over the stages
    :param states: List of all possible states as 3x2 numpy arrays
    :param label: str with type of game being played (racer types)
    :param number: int plot number
    """
    timestamps = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']
    fontsize = 15

    # Extract player positions and velocities over time from the states played
    player1_positions = [states[state_idx][:, 0] for state_idx in
                         states_played]  # (distance, lane, velocity) for player 1
    player2_positions = [states[state_idx][:, 1] for state_idx in
                         states_played]  # (distance, lane, velocity) for player 2

    player1_distances = [state[0] for state in player1_positions]
    player1_lanes = [state[1] for state in player1_positions]
    player1_velocities = [state[2] for state in player1_positions]

    player2_distances = [state[0] for state in player2_positions]
    player2_lanes = [state[1] for state in player2_positions]
    player2_velocities = [state[2] for state in player2_positions]

    # Create a plot with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plt.subplots_adjust(left=0.3, hspace=0.5)

    # Plot player positions (distance and lane)
    axs[0].plot(player1_distances, player1_lanes, label='Player 1', marker='o', linestyle='-', color='#3b719f')
    axs[0].plot(player2_distances, player2_lanes, label='Player 2', marker='s', linestyle='-', color='#e64345')
    # axs[0].invert_yaxis()

    # Ensure there is some minimum range for the axes
    axs[0].set_xlim(-0.1, len(states_played)-0.9)
    axs[0].set_ylim(max(player1_lanes + player2_lanes)+0.1,
                    min(player1_lanes + player2_lanes)-0.2)
    axs[0].xaxis.set_tick_params(labelsize=fontsize)
    axs[0].yaxis.set_tick_params(labelsize=fontsize)

    offset1 = 0
    offset2 = 0
    last_coords1 = ()
    last_coords2 = ()
    for i in range(len(player1_positions)):
        txt = timestamps[i]

        if last_coords1 == (player1_distances[i], player1_lanes[i]):
            offset1 += 0.07
        else:
            offset1 = 0
        axs[0].annotate(txt, (player1_distances[i], player1_lanes[i]),
                        (player1_distances[i] + offset1, player1_lanes[i] + 0.22), color='#3b719f', fontsize=fontsize)

        if last_coords2 == (player2_distances[i], player2_lanes[i]):
            offset2 += 0.07
        else:
            offset2 = 0
        axs[0].annotate(txt, (player2_distances[i], player2_lanes[i]),
                        (player2_distances[i] + offset2, player2_lanes[i] - 0.08), color='#e64345', fontsize=fontsize)

        last_coords1 = (player1_distances[i], player1_lanes[i])
        last_coords2 = (player2_distances[i], player2_lanes[i])

    axs[0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0].set_xlabel('Distance', fontsize=fontsize)
    axs[0].set_ylabel('Lane', fontsize=fontsize)
    axs[0].set_title('Race Position Overhead View', fontsize=fontsize)
    axs[0].legend(loc='center right', fontsize=fontsize)

    axs[0].grid(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # Plot player velocities
    axs[1].plot(player1_velocities, label='Player 1', marker='o', linestyle='-', color='#3b719f')
    axs[1].plot(player2_velocities, label='Player 2', marker='s', linestyle='-', color='#e64345')

    # Ensure there is some minimum range for the axes
    axs[1].set_xlim(-0.1, len(states_played)-0.9)
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].xaxis.set_tick_params(labelsize=fontsize)
    axs[1].yaxis.set_tick_params(labelsize=fontsize)

    axs[1].yaxis.set_major_locator(MultipleLocator(1))
    axs[1].set_xlabel('Stage', fontsize=fontsize)
    axs[1].set_ylabel('Velocity', fontsize=fontsize)
    axs[1].set_title('Player Velocities for Each Stage', fontsize=fontsize)
    axs[1].legend(loc='center right', fontsize=fontsize)

    axs[1].grid(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    # Show the plot
    plt.tight_layout()
    plt.savefig("plots/"+result_directory+"/Race_Visuals/" + label + "/" + str(number) + ".png")
    # plt.show()
    plt.close()

def plot_average_cost(average_cost, label, result_directory):
    '''
    Plots the costs for each player with respect to safety and rank
    :param average_cost: array of cost values
    :param label: str race type
    '''

    fontsize = 15
    # Flatten the array to a 1D array
    costs = average_cost.flatten()

    # Define labels for each bar
    labels = ['P1 Rank', 'P1 Safety', 'P2 Rank', 'P2 Safety']

    # Plot the bar graph
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, costs, color=['#e64345', '#6ba547', '#e64345', '#6ba547'])  # Using muted colors
    plt.subplots_adjust(bottom=0.15)

    ax.set_xlabel('Player Objectives', fontsize=fontsize)
    ax.set_ylabel('Average Cost Per Game', fontsize=fontsize)
    # ax.set_title('Average Cost- ' + label)

    ax.set_ylim(0, 14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    fig.savefig("plots/"+result_directory+"/Average_Costs/" + label + ".png")
    # plt.show()
    plt.close()


def plot_pareto_front(average_cost, labels, result_directory,):
    """
    Plot outcomes for all race types on cost space
    :param average_cost: array of outcomes for each player with respect to each objective
    :param labels: str race type
    """
    fontsize = 20
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    markers = [",", "o", "v", "^", "<", ">"]

    for i, costs in enumerate(average_cost):
        point1 = costs[0]
        point2 = costs[1]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.scatter(point1[0], point1[1], color=color, label=labels[i], marker=marker)
        plt.annotate('P1', (point1[0], point1[1]), textcoords="offset points", xytext=(fontsize, 0),
                     ha='center', fontsize=fontsize)
        plt.scatter(point2[0], point2[1], color=color,  marker=marker)
        plt.annotate('P2', (point2[0], point2[1]), textcoords="offset points", xytext=(fontsize, 0),
                     ha='center', fontsize=fontsize)

    # Remove duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # plt.title('Multi-Scenario Pareto Frontier')
    plt.xlabel('Average Rank Cost Per Game', fontsize=fontsize)
    plt.ylabel('Average Safety Cost Per Game', fontsize=fontsize)
    plt.legend(by_label.values(), by_label.keys(), loc='lower left',
               title="Scenario (Player 1-Player 2)", fontsize=15, title_fontsize=15)

    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # plt.ylim(3, 9)
    # plt.xlim(3, 9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig("plots/"+result_directory+"/Pareto_Fronts/" + "pareto.png")
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # Example data (to be replaced with actual data from your simulation)
    # Assuming the states are numpy arrays of the form: np.array([[dist1, lane1, vel1], [dist2, lane2, vel2]])
    # states = [
    #     np.array([[0, 0],
    #               [0, 1],
    #               [0, 0]]),
    #     np.array([[0, 1],
    #               [0, 0],
    #               [0, 1]]),
    #     np.array([[0, 2],
    #               [0, 0],
    #               [0, 1]]),
    #     np.array([[0, 2],
    #               [0, 0],
    #               [0, 1]])]
    #
    # # Example states_played (indices into the states list)
    # states_played = [0, 0, 1, 2, 3]
    #
    # # Call the plot function
    # plot_race(states_played, states)

    # average_costs = np.array([[10, 20], [4,7]])
    # plot_average_cost(average_costs, 'test')

    pair_labels = ["Aggressive-Aggressive", "Conservative-Conservative",
                   "Moderate-Moderate", "Moderate-Conservative",
                   "Moderate-Aggressive", "Conservative-Aggressive"]
    data = np.load("../offline_calcs/results.npz", allow_pickle=True)
    plot_pareto_front(data['values'], pair_labels)
