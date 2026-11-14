import gymnasium as gym

env = gym.make("CartPole-v1")

for episode in range(5):
    state, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample()  # random action: 0 = left, 1 = right
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode + 1}: total reward = {total_reward}")

env.close()