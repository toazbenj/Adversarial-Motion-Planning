import pygame
import sys
from course import Course

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 700, 700
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

def main():
    center_x, center_y = WIDTH // 2, HEIGHT // 2  # Center of the track

    course = Course(center_x, center_y)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bicycle Dynamics Simulation")

    # Clock for controlling the frame rate
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update the simulation
        course.update()

        # Draw everything
        screen.fill(WHITE)
        course.draw(screen)
        pygame.display.flip()

        clock.tick(60)  # Limit frame rate

if __name__ == "__main__":
    main()