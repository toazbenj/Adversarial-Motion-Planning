import pygame
import sys
from simulation import Simulation

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 700, 700
WHITE = (255, 255, 255)

def main():
    center_x, center_y = WIDTH // 2, HEIGHT // 2  # Center of the track

    simulation = Simulation(center_x, center_y)
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
        simulation.update()

        # Draw everything
        screen.fill(WHITE)
        simulation.draw(screen)
        pygame.display.flip()

        clock.tick(60)  # Limit frame rate

if __name__ == "__main__":
    main()