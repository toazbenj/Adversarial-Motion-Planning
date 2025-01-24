import pygame
from bicycle import Bicycle

# Colors
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)

class Course:
    def __init__(self, center_x, center_y, outer_radius=300, inner_radius=125):
        self.count = 0

        self.center_x = center_x
        self.center_y = center_y

        self.outer_radius = outer_radius
        self.inner_radius = inner_radius

        # self.bike1 = Bicycle(self, x=center_x + outer_radius - 110, y=center_y + 10)  # Initialize bike
        # self.bike2 = Bicycle(self, x=center_x + outer_radius - 90, y=center_y + 10, color=GREEN)  # Initialize bike

        self.bike1 = Bicycle(self, x=center_x + inner_radius +10, y=center_y)  # Initialize bike
        self.bike2 = Bicycle(self, x=center_x + inner_radius + 10, y=center_y - 50, color=GREEN, velocity_limit=20)  # Initialize bike

    def draw(self, screen):
        # Draw the racecourse
        # Fill background
        screen.fill(DARK_GREEN)

        # Draw the outer circle
        pygame.draw.circle(screen, GRAY, (self.center_x, self.center_y), self.outer_radius)

        # Draw the inner circle (cutout)
        pygame.draw.circle(screen, DARK_GREEN, (self.center_x, self.center_y), self.inner_radius)

        # Draw boundaries
        pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.outer_radius, 3)
        pygame.draw.circle(screen, BLACK, (self.center_x, self.center_y), self.inner_radius, 3)

        # Draw the bicycle
        self.bike1.draw(screen)
        self.bike2.draw(screen)


    def update(self):
        self.bike1.update_choices(self.count, self.bike2)
        self.bike2.update_choices(self.count, self.bike1)
        self.bike1.update_action(self.count)
        self.bike2.update_action(self.count)

        self.count += 1
