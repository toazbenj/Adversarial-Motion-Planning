from math import cos, sin, tan, atan2, radians, pi, degrees
import pygame

# cost weights
BOUNDS_WEIGHT = 1
COLLISION_WEIGHT = 10
DISTANCE_WEIGHT = -1 * 1/1000

GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class Trajectory:
    def __init__(self,  course, bike, color):
        self.points = []
        self.cost = 0
        self.color = color
        self.course = course
        self.bike = bike

        self.start_x = bike.x
        self.start_y = bike.y
        self.length = 0

        self.is_displaying = False

    def draw(self, screen, index=0):
        """
          Draw the trajectory and its associated cost on the screen.
          The index is used to space out cost text to avoid overlapping.
          """
        for pt in self.points:
            pygame.draw.circle(screen, self.color, (pt[0], pt[1]), 1)

        # Draw costs near trajectories with spacing adjustment
        if self.is_displaying:
            font = pygame.font.Font(None, 20)
            cost_text = font.render(f"{round(self.cost):.0f}", True, BLACK)
            text_x = self.points[-1][0] + 10
            text_y = self.points[-1][1] - 10
            screen.blit(cost_text, (text_x, text_y))

    def add_point(self, x, y):
        bounds_cost = BOUNDS_WEIGHT * self.check_bounds(x, y)
        if bounds_cost > 0:
            self.color = RED

        collision_cost = COLLISION_WEIGHT * self.check_collision(x, y)

        self.length = self.calc_arc_length_distance(x, y)
        distance_cost = DISTANCE_WEIGHT * self.length

        self.cost += round(bounds_cost + collision_cost + distance_cost, 2)
        self.points.append((round(x, 2), round(y, 2)))

    def check_bounds(self, new_x, new_y):
        # Calculate the distance from the center to the point
        distance_squared = (new_x - self.course.center_x) ** 2 + (new_y - self.course.center_y) ** 2

        # Check if the distance is between the outer radius and inner radius
        if self.course.inner_radius ** 2 <= distance_squared <= self.course.outer_radius ** 2:
            return 0
        else:
            return 1

    def check_collision(self, new_x, new_y):
        return 0

    def arc_length(self, x, y):
        # Calculate angular position in radians
        theta = atan2(y - self.course.center_y, x - self.course.center_x)
        theta = (theta + 2 * pi) % (2 * pi)  # Normalize to [0, 2π)

        # Compute arc length
        arc_length = theta * self.course.outer_radius
        return arc_length

    def calc_arc_length_distance(self, x, y):
        # Calculate arc lengths for both points
        arc1 = self.arc_length(self.start_x, self.start_y)
        arc2 = self.arc_length(x, y)

        # Calculate the absolute distance, handling wraparound
        distance = abs(arc2 - arc1)
        if distance > pi * self.course.outer_radius:  # Adjust for crossing the start/finish line
            distance = 2 * pi * self.course.outer_radius - distance
        return distance
