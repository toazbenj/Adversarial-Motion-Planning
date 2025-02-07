from math import cos, sin, tan, atan2, radians, pi, degrees, sqrt
import pygame
import numpy as np
from trajectory import Trajectory
from itertools import product
from cost_adjust_utils import cost_adjustment

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
    def __init__(self, course, x=300, y=300, v=0, color=BLUE, phi=radians(90), b=0, velocity_limit=15,
                 is_vector_cost=False, opponent=None):
        self.bicycle_size = 20
        self.color = color

        self.x = x
        self.y = y
        self.v = v
        self.phi = phi
        self.b = b

        self.lr = 1
        self.lf = 1

        self.a = 0
        self.steering_angle = 0
        self.velocity_limit = velocity_limit

        self.past_trajectories = []  # Store past positions
        self.choice_trajectories = [] # upcoming possible traj
        self.action_choices = [] # sequences of actions to create all possible trajectories
        self.chosen_action_sequence = [] # sequence of actions to create chosen trajectory
        self.action_interval = 70
        self.mpc_horizon = 2

        self.course = course

        self.new_choices()
        self.is_vector_cost = is_vector_cost
        self.opponent = opponent
        self.cost_arr = None

    def dynamics(self, acc, steering, x_in, y_in, v_in, phi_in, b_in):
        # Update positions
        x_next = x_in + v_in * cos(phi_in + b_in) * DT
        y_next = y_in + v_in * sin(phi_in + b_in) * DT

        # Update heading angle
        phi_next = phi_in + (v_in / self.lr) * sin(b_in) * DT

        # Update velocity
        v_next = v_in + acc * DT
        # velocity limit
        if v_next > self.velocity_limit:
            v_next = self.velocity_limit
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
        pygame.draw.polygon(screen, self.color, points)

        # Draw the past trajectory
        if len(self.past_trajectories) > 1:
            for traj in self.past_trajectories:
                traj.draw(screen)

        for i, traj in enumerate(set(self.choice_trajectories)):
            # print(f"Trajectory {i}: Cost = {traj.cost}, Points = {traj.points[0]}, {traj.points[-1]}")
            traj.draw(screen)

    def update_choices(self, count, other_bike):
        if count % (self.action_interval * self.mpc_horizon) == 0:
            self.new_choices(other_bike)

    def update_action(self, count):
        # Periodically compute actions
        if count % (self.action_interval * self.mpc_horizon) == 0:
            # self.new_choices()
            self.compute_action()
        # switch actions after action interval elapses
        if count % self.action_interval == 0:
            self.a = self.chosen_action_sequence[0][0] * ACCELERATION_INCREMENT
            self.steering_angle = self.chosen_action_sequence[0][1] * STEERING_INCREMENT
            self.chosen_action_sequence.remove(self.chosen_action_sequence[0])

        # Update the bicycle state
        self.x, self.y, self.v, self.phi, self.b  = self.dynamics(self.a, self.steering_angle, self.x, self.y, self.v, self.phi, self.b)

    def build_arr(self, trajectories):
        size = len(ACTION_LST)**self.mpc_horizon
        cost_arr = np.zeros((size, size))

        for i, traj in enumerate(trajectories):
            cost_row = np.zeros((1, size))
            cost_row[0, :] = traj.total_cost

            for other_traj in traj.intersecting_trajectories:
                cost_row[0][other_traj.number] += traj.collision_weight

            cost_arr[i] = cost_row

        self.cost_arr = cost_arr
        # np.savez(str(self.color)+'scalar.npz', arr=self.cost_arr)


    def build_vector_arr(self, trajectories):
        size = len(ACTION_LST) ** self.mpc_horizon
        safety_cost_arr = np.zeros((size, size))
        distance_cost_arr = np.zeros((size, size))

        for i, traj in enumerate(trajectories):
            cost_row_distance = np.zeros((1, size))
            cost_row_safety = np.zeros((1, size))

            cost_row_distance[0, :] = traj.total_cost

            for other_traj in traj.intersecting_trajectories:
                cost_row_safety[0][other_traj.number] += traj.collision_weight

            safety_cost_arr[i] = cost_row_safety
            distance_cost_arr[i] = cost_row_distance

        self.cost_arr = cost_adjustment(distance_cost_arr, safety_cost_arr, self.opponent.cost_arr.transpose())
        np.savez('vector_A.npz', arr=distance_cost_arr)


    def compute_action(self):
        if self.is_vector_cost:
            self.build_vector_arr(self.choice_trajectories)
        else:
            self.build_arr(self.choice_trajectories)

        action_index = np.argmin(np.max(self.cost_arr, axis=1))

        chosen_traj = self.choice_trajectories[action_index]
        chosen_traj.color = self.color
        chosen_traj.is_displaying = False
        chosen_traj.is_chosen = True

        # update costs of last trajectory after other players picked
        if len(self.past_trajectories) > 0:
            self.past_trajectories[-1].update()
        self.past_trajectories.append(chosen_traj)
        self.choice_trajectories.remove(chosen_traj)
        self.chosen_action_sequence = self.action_choices[action_index]

    def new_choices(self, other_bike=None):
        # Precompute trajectories for visualization
        self.choice_trajectories = []
        self.action_choices = generate_combinations(ACTION_LST, self.mpc_horizon)

        count = 0
        for action_sequence in self.action_choices:
            traj = Trajectory(bike=self, course=self.course, color=YELLOW)
            x_temp, y_temp, v_temp, phi_temp, b_temp = self.x, self.y, self.v, self.phi, self.b
            for action in action_sequence:
                acc = action[0] * ACCELERATION_INCREMENT
                steering = action[1] * STEERING_INCREMENT

                for _ in range(self.action_interval):
                    x_temp, y_temp, v_temp, phi_temp, b_temp = self.dynamics(acc, steering, x_temp, y_temp, v_temp, phi_temp, b_temp)
                    traj.add_point(x_temp, y_temp)

            traj.number = count
            self.choice_trajectories.append(traj)
            count += 1

        # check how far away the opponent is
        if other_bike is not None:
            is_in_range = sqrt((self.x - other_bike.x) ** 2 + (self.y - other_bike.y) ** 2) < self.action_interval * self.mpc_horizon

        # allow traj to know possible collisions
        if other_bike is not None and len(other_bike.choice_trajectories) > 0 and is_in_range:
            for traj in self.choice_trajectories:
                if traj.is_collision_checked:
                    continue
                for other_traj in other_bike.choice_trajectories:
                    if other_traj.is_collision_checked:
                        continue
                    other_traj.collision_checked = True
                    traj.collision_checked = True
                    traj.trajectory_intersection_optimized(other_traj)

    def get_costs(self):
        distance, bounds, collision, total = 0, 0, 0, 0
        for traj in self.past_trajectories:
            distance += traj.distance_cost
            bounds += traj.bounds_cost
            collision += traj.collision_cost
            total += traj.total_cost

        return distance, bounds, collision, total
