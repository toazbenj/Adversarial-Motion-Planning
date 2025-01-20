from math import cos, sin, tan, atan2, radians, pi, degrees
import pygame
import numpy as np
from trajectory import Trajectory


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
        self.action_interval = 100


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

    def update(self, count):
        self.x, self.y, self.v, self.phi, self.b  = self.dynamics(self.a, self.steering_angle, self.x, self.y, self.v, self.phi, self.b)
