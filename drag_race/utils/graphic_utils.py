    plot_pareto_front(data['values'], pair_labels)
    data = np.load("../offline_calcs/results.npz", allow_pickle=True)
                   "Moderate-Aggressive", "Conservative-Aggressive"]
                   "Moderate-Moderate", "Moderate-Conservative",
    pair_labels = ["Aggressive-Aggressive", "Conservative-Conservative",

    # plot_average_cost(average_costs, 'test')
    # average_costs = np.array([[10, 20], [4,7]])

    # plot_race(states_played, states)
    # # Call the plot function
    #
    # states_played = [0, 0, 1, 2, 3]
    # # Example states_played (indices into the states list)
    #
    #               [0, 1]])]
    #               [0, 0],
    #     np.array([[0, 2],
    #               [0, 1]]),
    #               [0, 0],
    #     np.array([[0, 2],
    #               [0, 1]]),
    #               [0, 0],
    #     np.array([[0, 1],
    #               [0, 0]]),
    #               [0, 1],
    #     np.array([[0, 0],
    # states = [
    # Assuming the states are numpy arrays of the form: np.array([[dist1, lane1, vel1], [dist2, lane2, vel2]])
    # Example data (to be replaced with actual data from your simulation)
if __name__ == '__main__':


    plt.close()
    # plt.show()
    fig.savefig("plots/"+result_directory+"/Pareto_Fronts/" + "pareto.png")
    plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.xlim(3, 9)
    # plt.ylim(3, 9)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

               title="Scenario (Player 1-Player 2)", fontsize=15, title_fontsize=15)
    plt.legend(by_label.values(), by_label.keys(), loc='lower left',
    plt.ylabel('Average Safety Cost Per Game', fontsize=fontsize)
    plt.xlabel('Average Rank Cost Per Game', fontsize=fontsize)
    # plt.title('Multi-Scenario Pareto Frontier')

    by_label = dict(zip(labels, handles))
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels in the legend

                     ha='center', fontsize=fontsize)
        plt.annotate('P2', (point2[0], point2[1]), textcoords="offset points", xytext=(fontsize, 0),
        plt.scatter(point2[0], point2[1], color=color,  marker=marker)
                     ha='center', fontsize=fontsize)
        plt.annotate('P1', (point1[0], point1[1]), textcoords="offset points", xytext=(fontsize, 0),
        plt.scatter(point1[0], point1[1], color=color, label=labels[i], marker=marker)

        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        point2 = costs[1]
        point1 = costs[0]
    for i, costs in enumerate(average_cost):

    markers = [",", "o", "v", "^", "<", ">"]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    fig, ax = plt.subplots(figsize=(10, 10))
    fontsize = 20
    """
    :param labels: str race type
    :param average_cost: array of outcomes for each player with respect to each objective
    Plot outcomes for all race types on cost space
    """
def plot_pareto_front(average_cost, labels, result_directory,):


    plt.close()
    # plt.show()
    fig.savefig("plots/"+result_directory+"/Average_Costs/" + label + ".png")

    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    ax.set_xticklabels(labels)
    ax.set_xticks(x)
    ax.set_ylim(6, 12)

    # ax.set_title('Average Cost- ' + label)
    ax.set_ylabel('Average Cost Per Game', fontsize=fontsize)
    ax.set_xlabel('Player Objectives', fontsize=fontsize)

    plt.subplots_adjust(bottom=0.15)
    ax.bar(x, costs, color=['#e64345', '#6ba547', '#e64345', '#6ba547'])  # Using muted colors
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the bar graph

    labels = ['P1 Rank', 'P1 Safety', 'P2 Rank', 'P2 Safety']
    # Define labels for each bar

    costs = average_cost.flatten()
    # Flatten the array to a 1D array
    fontsize = 15

    '''
    :param label: str race type
    :param average_cost: array of cost values
    Plots the costs for each player with respect to safety and rank
    '''
def plot_average_cost(average_cost, label, result_directory):

    plt.close()
    # plt.show()
    plt.savefig("plots/"+result_directory+"/Race_Visuals/" + label + "/" + str(number) + ".png")
    plt.tight_layout()
    # Show the plot
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].grid(False)

    axs[1].legend(loc='center right', fontsize=fontsize)
    axs[1].set_title('Player Velocities for Each Stage', fontsize=fontsize)
    axs[1].set_ylabel('Velocity', fontsize=fontsize)
    axs[1].set_xlabel('Stage', fontsize=fontsize)
    axs[1].yaxis.set_major_locator(MultipleLocator(1))

    axs[1].yaxis.set_tick_params(labelsize=fontsize)
    axs[1].xaxis.set_tick_params(labelsize=fontsize)
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_xlim(-0.1, len(states_played)-0.9)
    # Ensure there is some minimum range for the axes

    axs[1].plot(player2_velocities, label='Player 2', marker='s', linestyle='-', color='#e64345')
    axs[1].plot(player1_velocities, label='Player 1', marker='o', linestyle='-', color='#3b719f')
    # Plot player velocities

    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].grid(False)

    axs[0].legend(loc='center right', fontsize=fontsize)
    axs[0].set_title('Race Position Overhead View', fontsize=fontsize)
    axs[0].set_ylabel('Lane', fontsize=fontsize)
    axs[0].set_xlabel('Distance', fontsize=fontsize)
    axs[0].yaxis.set_major_locator(MultipleLocator(1))

        last_coords2 = (player2_distances[i], player2_lanes[i])
        last_coords1 = (player1_distances[i], player1_lanes[i])

                        (player2_distances[i] + offset2, player2_lanes[i] - 0.08), color='#e64345', fontsize=fontsize)
        axs[0].annotate(txt, (player2_distances[i], player2_lanes[i]),
            offset2 = 0
        else:
            offset2 += 0.07
        if last_coords2 == (player2_distances[i], player2_lanes[i]):

                        (player1_distances[i] + offset1, player1_lanes[i] + 0.22), color='#3b719f', fontsize=fontsize)
        axs[0].annotate(txt, (player1_distances[i], player1_lanes[i]),
            offset1 = 0
        else:
            offset1 += 0.07
        if last_coords1 == (player1_distances[i], player1_lanes[i]):

        txt = timestamps[i]
    for i in range(len(player1_positions)):
    last_coords2 = ()
    last_coords1 = ()
    offset2 = 0
    offset1 = 0

    axs[0].yaxis.set_tick_params(labelsize=fontsize)
    axs[0].xaxis.set_tick_params(labelsize=fontsize)
                    min(player1_lanes + player2_lanes)-0.2)
    axs[0].set_ylim(max(player1_lanes + player2_lanes)+0.1,
    axs[0].set_xlim(-0.1, len(states_played)-0.9)
    # Ensure there is some minimum range for the axes

    # axs[0].invert_yaxis()
    axs[0].plot(player2_distances, player2_lanes, label='Player 2', marker='s', linestyle='-', color='#e64345')
    axs[0].plot(player1_distances, player1_lanes, label='Player 1', marker='o', linestyle='-', color='#3b719f')
    # Plot player positions (distance and lane)

    plt.subplots_adjust(left=0.3, hspace=0.5)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    # Create a plot with two subplots

    player2_velocities = [state[2] for state in player2_positions]
    player2_lanes = [state[1] for state in player2_positions]
    player2_distances = [state[0] for state in player2_positions]

    player1_velocities = [state[2] for state in player1_positions]
    player1_lanes = [state[1] for state in player1_positions]
    player1_distances = [state[0] for state in player1_positions]

                         states_played]  # (distance, lane, velocity) for player 2
    player2_positions = [states[state_idx][:, 1] for state_idx in
                         states_played]  # (distance, lane, velocity) for player 1
    player1_positions = [states[state_idx][:, 0] for state_idx in
    # Extract player positions and velocities over time from the states played

    fontsize = 15
    timestamps = ['t0 ', 't1 ', 't2 ', 't3 ', 't4 ', 't5 ', 't6 ', 't7 ', 't8 ']
    """
    :param number: int plot number
    :param label: str with type of game being played (racer types)
    :param states: List of all possible states as 3x2 numpy arrays
    :param states_played: List of indices of states played over the stages

    and their velocities on a separate graph, with each data point labeled with a timestamp.
    Visualize the race game by plotting the positions (distance and lane) of the players over time
    """
def plot_race_view(states_played, states, label, result_directory, number):

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np

"""
Functions to graph state output and values charts for drag race game

Graphics
"""
