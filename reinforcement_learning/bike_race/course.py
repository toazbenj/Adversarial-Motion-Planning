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

        self.bike1 = Bicycle(self, x=center_x + inner_radius + 10, y=center_y)  # Initialize bike
        self.bike2 = Bicycle(self, x=center_x + inner_radius + 10, y=center_y-100, color=GREEN, is_vector_cost=True, velocity_limit=20, opponent=self.bike1)  # Initialize bike

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

        distance_cost1, bounds_cost1, collision_cost1, total_costs1 = self.bike1.get_costs()
        distance_cost2, bounds_cost2, collision_cost2, total_costs2 = self.bike2.get_costs()

        # display running costs

        # Display running costs for each bike
        font = pygame.font.Font(None, 36)  # Font and size

        # Bike 1 costs
        text_bike1 = font.render("Bike 1 Costs:", True, WHITE)
        text_bike1_bounds = font.render(f"Bounds Cost: {int(bounds_cost1)}", True, WHITE)
        text_bike1_distance = font.render(f"Distance Cost: {int(distance_cost1)}", True, WHITE)
        text_bike1_collision = font.render(f"Collision Cost: {int(collision_cost1)}", True, WHITE)
        text_bike1_total = font.render(f"Total Cost: {int(total_costs1)}", True, WHITE)

        # Bike 2 costs
        text_bike2 = font.render("Bike 2 Costs:", True, WHITE)
        text_bike2_bounds = font.render(f"Bounds Cost: {int(bounds_cost2)}", True, WHITE)
        text_bike2_distance = font.render(f"Distance Cost: {int(distance_cost2)}", True, WHITE)
        text_bike2_collision = font.render(f"Collision Cost: {int(collision_cost1)}", True, WHITE)
        text_bike2_total = font.render(f"Total Cost: {int(total_costs2)}", True, WHITE)

        # Draw the costs on the right side of the screen
        margin = 20
        start_y = margin
        screen_width = screen.get_width()

        # Draw bike 1 costs
        screen.blit(text_bike1, (screen_width - 300, start_y))
        screen.blit(text_bike1_bounds, (screen_width - 300, start_y + 40))
        screen.blit(text_bike1_distance, (screen_width - 300, start_y + 80))
        screen.blit(text_bike1_collision, (screen_width - 300, start_y + 120))
        screen.blit(text_bike1_total, (screen_width - 300, start_y + 160))

        # Draw bike 2 costs
        screen.blit(text_bike2, (screen_width - 300, start_y + 240))
        screen.blit(text_bike2_bounds, (screen_width - 300, start_y + 280))
        screen.blit(text_bike2_distance, (screen_width - 300, start_y + 320))
        screen.blit(text_bike2_collision, (screen_width - 300, start_y + 360))
        screen.blit(text_bike2_total, (screen_width - 300, start_y + 400))

        # Draw the bicycle
        self.bike1.draw(screen)
        self.bike2.draw(screen)


    def update(self):
        self.bike1.update_choices(self.count, self.bike2)
        self.bike2.update_choices(self.count, self.bike1)

        self.bike1.update_action(self.count)
        self.bike2.update_action(self.count)

        self.count += 1
