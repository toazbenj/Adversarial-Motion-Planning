import numpy as np
import gym
import pygame
from pygame.locals import QUIT
from time import sleep

# Initialize the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Q-Learning parameters
num_states = (1, 1, 6, 12)  # Discretization of observation space (position, velocity, angle, angular velocity)
num_actions = env.action_space.n
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.5  # Exploration rate
num_episodes = 10000  # Total episodes for training

# Discretization function for continuous states
def discretize_state(observation, bins):
    """Discretize continuous state space into a finite number of bins."""
    low = env.observation_space.low
    high = env.observation_space.high
    high[1] = 1.0  # Limit velocity
    high[3] = 2.0  # Limit angular velocity
    low[1] = -1.0
    low[3] = -2.0
    ratios = (observation - low) / (high - low)
    new_obs = np.clip((ratios * bins).astype(int), 0, [b - 1 for b in bins])
    return tuple(new_obs)

# Initialize Q-table
q_table = np.zeros(num_states + (num_actions,))

# Q-Learning algorithm
for episode in range(num_episodes):
    state = discretize_state(env.reset()[0], num_states)
    total_reward = 0

    done = False
    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take the action and observe the result
        next_observation, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_observation, num_states)

        # Update Q-value using the Q-Learning equation
        best_next_action = np.argmax(q_table[next_state])
        q_table[state + (action,)] += alpha * (
            reward + gamma * q_table[next_state + (best_next_action,)] - q_table[state + (action,)]
        )

        # Update state
        state = next_state
        total_reward += reward

    # Print episode summary
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Pygame visualization
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("CartPole Visualization")
clock = pygame.time.Clock()

state = discretize_state(env.reset()[0], num_states)
done = False

while not done:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            env.close()
            done = True

    # Get action and step environment
    action = np.argmax(q_table[state])
    next_observation, _, done, _, _ = env.step(action)
    state = discretize_state(next_observation, num_states)

    # Get the rendered frame
    frame = env.render()

    # Convert the frame to a Pygame surface
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))

    # Display the frame
    screen.blit(pygame.transform.scale(frame_surface, (600, 400)), (0, 0))
    pygame.display.flip()
    clock.tick(30)  # Limit to 30 FPS
    clock.tick_busy_loop(30)

print('done')