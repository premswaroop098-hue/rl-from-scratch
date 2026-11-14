import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from algos.ppo import PPOAgent

SEED = 0
TOTAL_STEPS = 150_000
ROLLOUT = 2048          # steps collected before each update


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)

    state, _ = env.reset(seed=SEED)
    ep_reward, ep_count, steps = 0.0, 0, 0
    records = []

    while steps < TOTAL_STEPS:
        states, actions, logps, rewards, dones, values = [], [], [], [], [], []

        # ---- 1) collect a rollout with the current policy ----
        for _ in range(ROLLOUT):
            action, logp, value = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # time-limit cutoff is not a real failure: bootstrap its value
            train_reward = reward
            if truncated and not terminated:
                train_reward += agent.gamma * agent.value(next_state)

            states.append(state)
            actions.append(action)
            logps.append(logp)
            rewards.append(train_reward)
            dones.append(float(done))
            values.append(value)

            state = next_state
            steps += 1

            if done:
                records.append({"episode": ep_count, "step": steps, "reward": ep_reward})
                ep_count += 1
                ep_reward = 0.0
                state, _ = env.reset()

        # ---- 2) learn from it ----
        last_value = agent.value(state)
        _, _, entropy = agent.update(states, actions, logps, rewards,
                                     dones, values, last_value)

        recent = [r["reward"] for r in records[-10:]]
        avg = np.mean(recent) if recent else float("nan")
        print(f"Steps {steps:7d} | recent avg reward {avg:6.1f} | entropy {entropy:.3f}")

    df = pd.DataFrame(records)
    df["reward_ma"] = df["reward"].rolling(20, min_periods=1).mean()
    df.to_csv("results/ppo_cartpole.csv", index=False)
    torch.save(agent.model.state_dict(), "results/ppo_cartpole.pt")

    plt.plot(df["step"], df["reward"], alpha=0.3, label="episode reward")
    plt.plot(df["step"], df["reward_ma"], label="20-episode average")
    plt.xlabel("Environment steps")
    plt.ylabel("Reward")
    plt.title("PPO on CartPole-v1")
    plt.legend()
    plt.savefig("results/ppo_cartpole.png", dpi=150)
    print("Saved results/ppo_cartpole.csv, .png and .pt")


if __name__ == "__main__":
    main()
