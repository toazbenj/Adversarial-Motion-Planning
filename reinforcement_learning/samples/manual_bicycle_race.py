import pygame
import sys
import math
from math import cos, sin, tan, atan2, radians

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
dt = 0.1  # Time step
steering_increment = radians(0.1)  # Increment for steering angle
acceleration_increment = 1  # Increment for acceleration

# Function to draw the racecourse
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

# Function to handle keyboard input
def handle_keys():
    global a, steering_angle
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:  # Accelerate
        a = acceleration_increment
    if keys[pygame.K_s]:  # Decelerate
        a = -acceleration_increment
    if keys[pygame.K_a]:  # Turn left
        steering_angle -= steering_increment
    if keys[pygame.K_d]:  # Turn right
        steering_angle += steering_increment
    if keys[pygame.K_SPACE]:
        a = 0
        steering_angle = 0

# Function to update the bicycle's position and dynamics
def update_bicycle():
    global x, y, v, phi, b, steering_angle
    keys = pygame.key.get_pressed()

    # Update positions
    x += v * cos(phi + b) * dt
    y += v * sin(phi + b) * dt

    # Update heading angle
    phi += (v / lr) * sin(b) * dt

    # Update velocity
    v += a * dt
    # velocity limit
    if v > VELOCITY_LIMIT:
        v = VELOCITY_LIMIT
    v = max(0, v)  # Prevent negative velocity

    # Update velocity angle
    # Limit steering angle dynamically
    # if abs(steering_angle) > STEER_LIMIT:
    #     steering_angle = max(-STEER_LIMIT, min(STEER_LIMIT, steering_angle))
    # # Adjust the steering angle relative to the current orientation
    # adjusted_steering = steering_angle + phi

    # b = atan2(lr * tan(adjusted_steering), lr + lf)

    b = atan2(lr * tan(steering_angle), lr + lf)


    if keys[pygame.K_SPACE]:
        v = 0
        b = 0

# Function to draw the bicycle (blue square)
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


# Main loop
def main():
    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Handle input and update dynamics
        handle_keys()
        update_bicycle()

        # Draw everything
        draw_racecourse()
        draw_bicycle()

        # Update display
        pygame.display.flip()

        # Limit frame rate
        clock.tick(60)

if __name__ == "__main__":
    main()
