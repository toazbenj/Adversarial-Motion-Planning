import pygame
from bicycle import Bicycle

# Colors
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
BLACK = (0, 0, 0)

class Simulation:
    def __init__(self, center_x, center_y, outer_radius=300, inner_radius=150):
        self.bicycle = Bicycle(x=300, y=300)  # Initialize bike
        self.count = 0

        self.center_x = center_x
        self.center_y = center_y

        self.outer_radius = outer_radius
        self.inner_radius = inner_radius

    def draw(self, screen):
        # Draw the racecourse
        # Fill background
        screen.fill(WHITE)

        # Draw the outer circle
        pygame.draw.circle(screen, GRAY, (self.center_x, self.center_y), self.outer_radius)

        # Draw the inner circle (cutout)
        pygame.draw.circle(screen, WHITE, (self.center_x, self.center_y), self.inner_radius)

        # Draw boundaries
        pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.outer_radius, 3)
        pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.inner_radius, 3)

        # Draw the bicycle
        self.bicycle.draw(screen)

    def update(self):
        self.bicycle.update(self.count)
        self.count += 1
