from functools import lru_cache
from math import cos, sin, tan, atan2, radians, pi, degrees
import pygame
import numpy as np

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

ACTION_LST = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
DT = 0.05  # Time step
STEERING_INCREMENT = radians(1)  # Increment for steering angle
ACCELERATION_INCREMENT = 3  # Increment for acceleration
STEER_LIMIT = radians(20)
VELOCITY_LIMIT = 15


class Bicycle:
    def __init__(self, x=300, y=300, v=0, phi=radians(90), b=0):
        self.x = x
        self.y = y
        self.v = v
        self.phi = phi
        self.b = b

        self.lr = 1
        self.lf = 1

        self.a = 0
        self.steering_angle = 0

        self.past_trajectory = []  # Store past positions
        self.choice_trajectories = []
        self.costs = []
        self.action_interval = 50

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
        bicycle_size = 20
        points = [
            (self.x + bicycle_size * cos(self.phi) - bicycle_size / 2 * sin(self.phi),
             self.y + bicycle_size * sin(self.phi) + bicycle_size / 2 * cos(self.phi)),
            (self.x - bicycle_size * cos(self.phi) - bicycle_size / 2 * sin(self.phi),
             self.y - bicycle_size * sin(self.phi) + bicycle_size / 2 * cos(self.phi)),
            (self.x - bicycle_size * cos(self.phi) + bicycle_size / 2 * sin(self.phi),
             self.y - bicycle_size * sin(self.phi) - bicycle_size / 2 * cos(self.phi)),
            (self.x + bicycle_size * cos(self.phi) + bicycle_size / 2 * sin(self.phi),
             self.y + bicycle_size * sin(self.phi) - bicycle_size / 2 * cos(self.phi))
        ]
        pygame.draw.polygon(screen, BLUE, points)

        # Draw the past trajectory
        if len(self.past_trajectory) > 1:
            pygame.draw.lines(screen, (0, 0, 255), False, self.past_trajectory, 2)  # Blue line

        self.draw_trajectories(screen)

    def update(self, count):
        # Periodically compute actions
        if count % self.action_interval == 0:
            self.compute_action()

        # Update the bicycle state
        self.x, self.y, self.v, self.phi, self.b  = self.dynamics(self.a, self.steering_angle, self.x, self.y, self.v, self.phi, self.b)

    def compute_action(self):
        self.costs = self.action_costs()
        action_index = np.argmin(self.costs)
        action = ACTION_LST[action_index]

        # Update the bicycle's control inputs
        self.a = action[0] * ACCELERATION_INCREMENT
        self.steering_angle = action[1] * STEERING_INCREMENT

        # Precompute trajectories for visualization
        self.choice_trajectories = []
        for action in ACTION_LST:
            x_temp, y_temp, v_temp, phi_temp, b_temp = self.x, self.y, self.v, self.phi, self.b
            for _ in range(self.action_interval):
                x_temp, y_temp, v_temp, phi_temp, b_temp = self.dynamics(action[0], action[1], x_temp, y_temp, v_temp, phi_temp, b_temp)
            self.choice_trajectories.append((x_temp, y_temp))
