import sys
sys.path.append(".")

import gymnasium as gym
from algos.dqn import DQNAgent

env = gym.make("CartPole-v1")
agent = DQNAgent(state_dim=4, action_dim=2)

# fill the buffer with random experience
state, _ = env.reset(seed=0)
for _ in range(500):
    action = agent.act(state, epsilon=1.0)
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.buffer.add(state, action, reward, next_state, float(terminated))
    state = next_state
    if terminated or truncated:
        state, _ = env.reset()

print("Buffer size:", len(agent.buffer))

# run some learning steps
losses = [agent.update() for _ in range(200)]
print("First loss:", round(losses[0], 4))
print("Last loss: ", round(losses[-1], 4))

# greedy action from the network
print("Greedy action:", agent.act(state, epsilon=0.0))
