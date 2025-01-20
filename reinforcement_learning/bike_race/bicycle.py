from math import cos, sin, tan, atan2, radians, pi, degrees
import pygame
import numpy as np
from trajectory import Trajectory
from itertools import product


RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

ACTION_LST = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
DT = 0.05  # Time step
STEERING_INCREMENT = radians(1.5)  # Increment for steering angle
ACCELERATION_INCREMENT = 3  # Increment for acceleration
STEER_LIMIT = radians(20)
VELOCITY_LIMIT = 15


def generate_combinations(numbers, num_picks):
    """
    Generate all combinations of choices by picking `num_picks` times from the list `numbers`.

    Args:
        numbers (list): The list of numbers to pick from.
        num_picks (int): The number of times to pick.

    Returns:
        list of tuples: All combinations of length `num_picks`.
    """
    if not numbers or num_picks <= 0:
        return []

    # Use itertools.product to generate combinations
    combinations = list(product(numbers, repeat=num_picks))
    combinations = [list(element) for element in combinations]
    return combinations


class Bicycle:
    def __init__(self, course, x=300, y=300, v=0, phi=radians(90), b=0):
        self.bicycle_size = 20

        self.x = x
        self.y = y
        self.v = v
        self.phi = phi
        self.b = b

        self.lr = 1
        self.lf = 1

        self.a = 0
        self.steering_angle = 0

        self.past_trajectories = []  # Store past positions
        self.choice_trajectories = [] # upcoming possible traj
        self.action_choices = [] # sequences of actions to create all possible trajectories
        self.chosen_action_sequence = [] # sequence of actions to create chosen trajectory
        self.action_interval = 50
        self.mpc_horizon = 2

        self.course = course

        self.new_choices()

    def dynamics(self, acc, steering, x_in, y_in, v_in, phi_in, b_in):
        # Update positions
        x_next = x_in + v_in * cos(phi_in + b_in) * DT
        y_next = y_in + v_in * sin(phi_in + b_in) * DT

        # Update heading angle
        phi_next = phi_in + (v_in / self.lr) * sin(b_in) * DT

        # Update velocity
        v_next = v_in + acc * DT
        # velocity limit
        if v_next > VELOCITY_LIMIT:
            v_next = VELOCITY_LIMIT
        v_next = max(0, v_next)  # Prevent negative velocity

        b_next = atan2(self.lr * tan(steering), self.lr + self.lf)

        return x_next, y_next, v_next, phi_next, b_next


    def draw(self, screen):
        # Draw the bike
        points = [
            (self.x + self.bicycle_size * cos(self.phi) - self.bicycle_size / 2 * sin(self.phi),
             self.y + self.bicycle_size * sin(self.phi) + self.bicycle_size / 2 * cos(self.phi)),
            (self.x - self.bicycle_size * cos(self.phi) - self.bicycle_size / 2 * sin(self.phi),
             self.y - self.bicycle_size * sin(self.phi) + self.bicycle_size / 2 * cos(self.phi)),
            (self.x - self.bicycle_size * cos(self.phi) + self.bicycle_size / 2 * sin(self.phi),
             self.y - self.bicycle_size * sin(self.phi) - self.bicycle_size / 2 * cos(self.phi)),
            (self.x + self.bicycle_size * cos(self.phi) + self.bicycle_size / 2 * sin(self.phi),
             self.y + self.bicycle_size * sin(self.phi) - self.bicycle_size / 2 * cos(self.phi))
        ]
        pygame.draw.polygon(screen, BLUE, points)

        # Draw the past trajectory
        if len(self.past_trajectories) > 1:
            for traj in self.past_trajectories:
                traj.draw(screen)

        for i, traj in enumerate(set(self.choice_trajectories)):
            print(f"Trajectory {i}: Cost = {traj.cost}, Points = {traj.points[0]}, {traj.points[-1]}")
            traj.draw(screen, index=i)


    def update(self, count):
        # Periodically compute actions
        if count % (self.action_interval * self.mpc_horizon) == 0:
            self.new_choices()
            self.compute_action()
        # switch actions after action interval elapses
        if count % self.action_interval == 0:
            self.a = self.chosen_action_sequence[0][0] * ACCELERATION_INCREMENT
            self.steering_angle = self.chosen_action_sequence[0][1] * STEERING_INCREMENT
            self.chosen_action_sequence.remove(self.chosen_action_sequence[0])

        # Update the bicycle state
        self.x, self.y, self.v, self.phi, self.b  = self.dynamics(self.a, self.steering_angle, self.x, self.y, self.v, self.phi, self.b)

    def compute_action(self):
        cost_arr = np.zeros(len(ACTION_LST)**self.mpc_horizon)
        for i, traj in enumerate(self.choice_trajectories):
            cost_arr[i] = traj.cost

        action_index = np.argmin(cost_arr)
        chosen_traj = self.choice_trajectories[action_index]
        chosen_traj.color = GREEN
        chosen_traj.is_displaying = False
        self.past_trajectories.append(chosen_traj)
        self.choice_trajectories.remove(chosen_traj)
        self.chosen_action_sequence = self.action_choices[action_index]

        # self.a = self.chosen_action_sequence[0][0] * ACCELERATION_INCREMENT
        # self.steering_angle =  self.chosen_action_sequence[0][1] * STEERING_INCREMENT
        # self.chosen_action_sequence.remove(self.chosen_action_sequence[0])

    def new_choices(self):
        # Precompute trajectories for visualization
        self.choice_trajectories = []
        self.action_choices = generate_combinations(ACTION_LST, self.mpc_horizon)

        for action_sequence in self.action_choices:
            traj = Trajectory(bike=self, course=self.course, color=YELLOW)
            x_temp, y_temp, v_temp, phi_temp, b_temp = self.x, self.y, self.v, self.phi, self.b
            for action in action_sequence:
                acc = action[0] * ACCELERATION_INCREMENT
                steering = action[1] * STEERING_INCREMENT

                for _ in range(self.action_interval):
                    x_temp, y_temp, v_temp, phi_temp, b_temp = self.dynamics(acc, steering, x_temp, y_temp, v_temp, phi_temp, b_temp)
                    traj.add_point(x_temp, y_temp)

            self.choice_trajectories.append(traj)

