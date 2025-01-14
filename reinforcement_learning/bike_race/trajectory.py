from math import cos, sin, tan, atan2, radians, pi, degrees
import pygame

BOUNDS_WEIGHT = 100
COLLISION_WEIGHT = 10
DISTANCE_WEIGHT = -1 * 1/50


def check_bounds(new_x, new_y, sim):
    # Calculate the distance from the center to the point
    distance_squared = (new_x - sim.center_x) ** 2 + (new_y - sim.center_y) ** 2

    # Check if the distance is between the outer radius and inner radius
    if sim.inner_radius ** 2 <= distance_squared <= sim.radius ** 2:
        return 0
    else:
        return 1


def check_collision(new_x, new_y):
    return 0


def calculate_angle(x_pos, y_pos, center_x, center_y):
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


def calc_radial_distance(x1, y1, x2, y2, center_x, center_y):
    final_angle = calculate_angle(x1, y1, center_x, center_y)
    initial_angle = calculate_angle(x2, y2, center_x, center_y)
    print(final_angle - initial_angle)
    return final_angle - initial_angle


def calculate_arc_length(x, y, center_x, center_y, radius):
    # Calculate angular position in radians
    theta = atan2(y - center_y, x - center_x)
    theta = (theta + 2 * pi) % (2 * pi)  # Normalize to [0, 2π)

    # Compute arc length
    arc_length = theta * radius
    return arc_length


def arc_length_distance(x1, y1, x2, y2, center_x, center_y, radius):
    # Calculate arc lengths for both points
    arc1 = calculate_arc_length(x1, y1, center_x, center_y, radius)
    arc2 = calculate_arc_length(x2, y2, center_x, center_y, radius)

    # Calculate the absolute distance, handling wraparound
    distance = abs(arc2 - arc1)
    if distance > pi * radius:  # Adjust for crossing the start/finish line
        distance = 2 * pi * radius - distance
    return distance


class Trajectory:
    def __init__(self):
        self.points = []
        self.cost = 0
        self.color = (0, 0, 255)

    def draw_trajectory(self, screen):
        for point in self.points:
            pygame.draw.circle(screen, self.color, point, 5)

        # Draw costs near trajectories
        font = pygame.font.Font(None, 24)
        cost_text = font.render(f"{self.cost:.2f}", True, self.color)
        screen.blit(cost_text, (self.points[-1][0] + 10, self.points[-1][0] - 10))

    def cost(self):
        for point in self.points:
            x, y = point

            bounds_cost = BOUNDS_WEIGHT * check_bounds(x, y)
            collision_cost = COLLISION_WEIGHT * check_collision(x, y)
            distance_cost = DISTANCE_WEIGHT * arc_length_distance(x, y)
            self.cost += bounds_cost + collision_cost + distance_cost
