import pygame
import sys
from math import cos, sin, tan, atan2, radians, pi, degrees
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 700, 700

STEER_LIMIT = radians(20)
VELOCITY_LIMIT = 30

# Colors
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bicycle Dynamics Simulation")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Circular track parameters
center_x, center_y = WIDTH // 2, HEIGHT // 2  # Center of the track
radius = 300  # Radius of the circular track
inner_radius = radius // 2  # Radius of the inner cutout

# Initial positions for the squares (middle-left of the track)
middle_left_x = center_x + radius
middle_left_y = center_y

# Bicycle dynamics variables
# x, y = WIDTH // 2, HEIGHT // 2
x, y = middle_left_x - 110,  middle_left_y - 15

v = 0  # Initial velocity
phi = radians(90)  # Heading angle
b = 0  # Velocity angle
lr, lf = 1, 1  # Bicycle parameters (lengths of rear/front axles)
steering_angle = 0  # Steering input
a = 0  # Acceleration

# Constants
DT = 0.1  # Time step
STEERING_INCREMENT = radians(0.1)  # Increment for steering angle
ACCELERATION_INCREMENT = 1  # Increment for acceleration

ACTION_LST = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
ACTION_INTERVAL = 10

BOUNDS_WEIGHT = 5
COLLISION_WEIGHT = 10
DISTANCE_WEIGHT = -3 * 1/25


def dynamics(acc, steering, x_in, y_in, v_in, phi_in, b_in):
    x_next, y_next, v_next, phi_next, b_next = x_in, y_in, v_in, phi_in, b_in

    # Update positions
    x_next += v_next * cos(phi_next + b_next) * DT
    y_next += v_next * sin(phi_next + b_next) * DT

    # Update heading angle
    phi_next += (v_next / lr) * sin(b_next) * DT

    # Update velocity
    v_next += acc * DT
    # velocity limit
    if v_next > VELOCITY_LIMIT:
        v_next = VELOCITY_LIMIT
    v_next= max(0, v_next)  # Prevent negative velocity

    b_next = atan2(lr * tan(steering), lr + lf)

    return x_next, y_next, v_next, phi_next, b_next


def check_bounds(new_x, new_y):
    # Calculate the distance from the center to the point
    distance_squared = (new_x - center_x) ** 2 + (new_y - center_y) ** 2

    # Check if the distance is between the outer radius and inner radius
    if inner_radius ** 2 <= distance_squared <= radius ** 2:
        return 0
    else:
        return 1


def check_collision(new_x, new_y):
    return 0


def calculate_angle(x_pos, y_pos):
    # Translate point to the circle's center
    dx = x_pos - center_x
    dy = y_pos - center_y

    # Calculate angle in radians
    angle_radians = atan2(dy, dx)

    # # Normalize angle to be in range [0, 2π)
    angle_radians = (angle_radians + 2 * pi) % (2 * pi)

    # Convert to degrees
    angle_degrees = degrees(angle_radians)

    return angle_degrees


def calc_radial_distance(new_x, new_y):
    final_angle = calculate_angle(new_x, new_y)
    initial_angle = calculate_angle(x, y)
    print(final_angle - initial_angle)
    return final_angle - initial_angle


from math import atan2, pi


def calculate_arc_length(x, y, center_x, center_y, radius):
    # Calculate angular position in radians
    theta = atan2(y - center_y, x - center_x)
    theta = (theta + 2 * pi) % (2 * pi)  # Normalize to [0, 2π)

    # Compute arc length
    arc_length = theta * radius
    return arc_length


def arc_length_distance(x2, y2):
    # Calculate arc lengths for both points
    arc1 = calculate_arc_length(x, y, center_x, center_y, radius)
    arc2 = calculate_arc_length(x2, y2, center_x, center_y, radius)

    # Calculate the absolute distance, handling wraparound
    distance = abs(arc2 - arc1)
    if distance > pi * radius:  # Adjust for crossing the start/finish line
        distance = 2 * pi * radius - distance
    return distance



def action_costs():
    cost_arr = np.zeros(len(ACTION_LST))

    for i, action in enumerate(ACTION_LST):
        x_next, y_next, v_next, phi_next, b_next = x, y, v, phi, b

        acc = action[0] * ACCELERATION_INCREMENT
        steering = action[1] * STEERING_INCREMENT

        for j in range(ACTION_INTERVAL):
            x_next, y_next, v_next, phi_next, b_next = dynamics(acc, steering,
                                                                x_next, y_next, v_next, phi_next, b_next)

        bounds_cost = BOUNDS_WEIGHT * check_bounds(x_next, y_next)
        collision_cost = COLLISION_WEIGHT * check_collision(x_next, y_next)
        distance_cost = DISTANCE_WEIGHT * arc_length_distance(x_next, y_next)

        cost_arr[i] = bounds_cost + collision_cost + distance_cost

    return cost_arr


def action_update(count):
    global a, steering_angle

    if count % ACTION_INTERVAL == 0:

        # evaluate action costs
        costs_arr = action_costs()
        # print(costs_arr)
        # pick action with least cost
        action_index = np.argmin(costs_arr)

        # update dynamics
        a = ACTION_LST[action_index][0] * ACCELERATION_INCREMENT
        steering_angle += ACTION_LST[action_index][1] * STEERING_INCREMENT


        # a_choice = int(random.random() * 3) - 1
        # phi_choice = int(random.random() * 3) - 1
        # a = a_choice * ACCELERATION_INCREMENT
        # steering_angle += phi_choice * STEERING_INCREMENT
    else:
        pass


def update_bicycle():
    global x, y, v, phi, b, steering_angle
    x, y, v, phi, b = dynamics(a, steering_angle, x, y, v, phi, b)


def draw_racecourse():
    # Fill background
    screen.fill(WHITE)

    # Draw the outer circle
    pygame.draw.circle(screen, GRAY, (center_x, center_y), radius)

    # Draw the inner circle (cutout)
    pygame.draw.circle(screen, WHITE, (center_x, center_y), inner_radius)

    # Draw boundaries
    pygame.draw.circle(screen, BLACK, (center_x, center_y), radius, 3)
    pygame.draw.circle(screen, BLACK, (center_x, center_y), inner_radius, 3)


def draw_bicycle():
    bicycle_size = 20
    # Calculate the rotated rectangle
    points = [
        (x + bicycle_size * cos(phi) - bicycle_size / 2 * sin(phi),
         y + bicycle_size * sin(phi) + bicycle_size / 2 * cos(phi)),
        (x - bicycle_size * cos(phi) - bicycle_size / 2 * sin(phi),
         y - bicycle_size * sin(phi) + bicycle_size / 2 * cos(phi)),
        (x - bicycle_size * cos(phi) + bicycle_size / 2 * sin(phi),
         y - bicycle_size * sin(phi) - bicycle_size / 2 * cos(phi)),
        (x + bicycle_size * cos(phi) + bicycle_size / 2 * sin(phi),
         y + bicycle_size * sin(phi) - bicycle_size / 2 * cos(phi))
    ]
    pygame.draw.polygon(screen, BLUE, points)


def main():
    count = 0
    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Handle input and update dynamics
        action_update(count)

        update_bicycle()

        # Draw everything
        draw_racecourse()
        draw_bicycle()

        # Update display
        pygame.display.flip()

        # Limit frame rate
        clock.tick(60)

        count += 1


if __name__ == "__main__":
    main()
