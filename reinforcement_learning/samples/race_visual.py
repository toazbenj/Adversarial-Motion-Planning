import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 700, 700

# Colors
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Circular Racecourse with Moving Squares")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Circular track parameters
center_x, center_y = WIDTH // 2, HEIGHT // 2  # Center of the track
radius = 250  # Radius of the circular track
inner_radius = radius//2  # Radius of the inner cutout
angle = 0  # Starting angle for circular motion (degrees)
angle_speed = 1  # Speed of rotation (degrees per frame)

# Initial positions for the squares (middle-left of the track)
middle_left_x = center_x + radius
middle_left_y = center_y
square2 = pygame.Rect(middle_left_x - 110, middle_left_y - 15, 30, 30)  # Blue square
red_square_size = 30

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

def handle_keys():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:  # Move up
        square2.y -= 5
    if keys[pygame.K_s]:  # Move down
        square2.y += 5
    if keys[pygame.K_a]:  # Move left
        square2.x -= 5
    if keys[pygame.K_d]:  # Move right
        square2.x += 5

def draw_squares():
    # Calculate the position of the red square using circular motion
    global angle
    offset = 40
    red_x = center_x + (radius - offset) * math.cos(math.radians(angle)) - red_square_size / 2
    red_y = center_y + (radius - offset) * math.sin(math.radians(angle)) - red_square_size / 2
    red_square = pygame.Rect(red_x, red_y, red_square_size, red_square_size)

    # Draw the red and blue squares
    pygame.draw.rect(screen, RED, red_square)
    pygame.draw.rect(screen, BLUE, square2)

    # Update the angle for the circular motion
    angle = (angle + angle_speed) % 360

def main():
    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Handle square movement
        handle_keys()

        # Draw the racecourse and squares
        draw_racecourse()
        draw_squares()

        # Update display
        pygame.display.flip()

        # Limit frame rate
        clock.tick(60)

if __name__ == "__main__":
    main()
