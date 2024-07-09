"""
Graphics

Function to graph state output from drag race game
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as plticker


def plot_race(states_played, states):
    """
    Visualize the race game by plotting the positions (distance and lane) of the players over time
    and their velocities on a separate graph, with each data point labeled with a timestamp.

    :param states_played: List of indices of states played over the stages
    :param states: List of all possible states as 3x2 numpy arrays
    """
    timestamps = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']

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
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(left=0.3, hspace=0.5)

    # Plot player positions (distance and lane)

    axs[0].plot(player1_distances, player1_lanes, label='Player 1', marker='o', linestyle='-')
    axs[0].plot(player2_distances, player2_lanes, label='Player 2', marker='s', linestyle='-')
    # axs[0].invert_yaxis()

    # Ensure there is some minimum range for the axes
    axs[0].set_xlim(-0.1, len(states_played)-0.9)
    axs[0].set_ylim(max(player1_lanes + player2_lanes)+0.1, min(player1_lanes + player2_lanes)-0.1)

    offset1 = 0.05
    offset2 = 0.05
    last_coords1 = ()
    last_coords2 = ()
    for i in range(len(player1_positions)):
        txt = timestamps[i]

        if last_coords1 == (player1_distances[i], player1_lanes[i]):
            offset1 += 0.1
        else:
            offset1 = 0.05
        axs[0].annotate(txt, (player1_distances[i], player1_lanes[i]),
                        (player1_distances[i] + offset1, player1_lanes[i] + 0.05))

        if last_coords2 == (player2_distances[i], player2_lanes[i]):
            offset2 += 0.1
        else:
            offset2 = 0.05
        axs[0].annotate(txt, (player2_distances[i], player2_lanes[i]),
                        (player2_distances[i] + offset2, player2_lanes[i] + 0.05))

        last_coords1 = (player1_distances[i], player1_lanes[i])
        last_coords2 = (player2_distances[i], player2_lanes[i])

    axs[0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0].set_xlabel('Distance')
    axs[0].set_ylabel('Lane')
    axs[0].set_title('Player Positions (Distance vs Lane)')
    axs[0].legend(loc='center right')
    axs[0].grid(False)

    # Plot player velocities
    axs[1].plot(player1_velocities, label='Player 1', marker='o', linestyle='-')
    axs[1].plot(player2_velocities, label='Player 2', marker='s', linestyle='-')

    # Ensure there is some minimum range for the axes
    axs[1].set_xlim(-0.1, len(states_played)-0.9)
    axs[1].set_ylim(min(player1_velocities + player2_velocities)-0.1, max(player1_velocities + player2_velocities)+0.1)

    axs[1].yaxis.set_major_locator(MultipleLocator(1))
    axs[1].set_xlabel('Stage')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Player Velocities over Time')
    axs[1].legend(loc='center right')
    axs[1].grid(False)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example data (to be replaced with actual data from your simulation)
    # Assuming the states are numpy arrays of the form: np.array([[dist1, lane1, vel1], [dist2, lane2, vel2]])
    states = [
        np.array([[0, 0],
                  [0, 1],
                  [0, 0]]),
        np.array([[0, 1],
                  [0, 0],
                  [0, 1]]),
        np.array([[0, 2],
                  [0, 0],
                  [0, 1]]),
        np.array([[0, 2],
                  [0, 0],
                  [0, 1]])]

    # Example states_played (indices into the states list)
    states_played = [0, 0, 1, 2, 3]

    # Call the plot function
    plot_race(states_played, states)
