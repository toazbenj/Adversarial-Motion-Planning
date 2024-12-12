import pygame
from pygame.locals import QUIT

# Initialize Pygame
pygame.init()

# Set up the display
window_width, window_height = 400, 300
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Pygame Box Example")

# Define box properties
box_color = (255, 0, 0)  # Red color
box_width, box_height = 50, 50
box_x, box_y = (window_width - box_width) // 2, (window_height - box_height) // 2  # Centered

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Fill the screen with a background color
    screen.fill((0, 0, 0))  # Black background

    # Draw the box
    pygame.draw.rect(screen, box_color, (box_x, box_y, box_width, box_height))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
