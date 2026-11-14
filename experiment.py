import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from algos.dqn import DQNAgent

SEEDS = [0, 1, 2, 3, 4]       # use [0, 1] first for a quick test
EPISODES = 400
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.985


def run(seed, double):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                     double=double)

    rewards = []
    for episode in range(EPISODES):
        epsilon = max(EPS_END, EPS_START * EPS_DECAY ** episode)
        state, _ = env.reset(seed=seed * 1000 + episode)
        total, done = 0.0, False

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.add(state, action, reward, next_state, float(terminated))
            agent.update()
            state = next_state
            total += reward

        rewards.append(total)
    env.close()
    return rewards


def main():
    rows = []
    for name, double in [("DQN", False), ("Double DQN", True)]:
        for seed in SEEDS:
            print(f"Running {name}, seed {seed} ...", flush=True)
            for ep, r in enumerate(run(seed, double)):
                rows.append({"algo": name, "seed": seed, "episode": ep, "reward": r})

    df = pd.DataFrame(rows)
    df.to_csv("results/comparison.csv", index=False)

    # learning curves: mean +/- std across seeds
    plt.figure(figsize=(8, 5))
    for name, group in df.groupby("algo"):
        wide = group.pivot(index="episode", columns="seed", values="reward")
        smooth = wide.rolling(20, min_periods=1).mean()
        mean, std = smooth.mean(axis=1), smooth.std(axis=1)
        plt.plot(mean.index, mean, label=name)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Reward (20-episode average)")
    plt.title(f"DQN vs Double DQN on CartPole-v1 ({len(SEEDS)} seeds)")
    plt.legend()
    plt.savefig("results/comparison.png", dpi=150)

    # final performance summary
    last = df[df.episode >= EPISODES - 50].groupby(["algo", "seed"])["reward"].mean()
    print("\nAverage reward over the last 50 episodes:")
    print(last.groupby("algo").agg(["mean", "std"]).round(1))


if __name__ == "__main__":
    main()
