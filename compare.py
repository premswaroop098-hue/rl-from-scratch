import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from algos.dqn import DQNAgent
from algos.ppo import PPOAgent

SEEDS = [0, 1, 2, 3, 4]       # for a quick test use [0] and TOTAL_STEPS = 20_000
TOTAL_STEPS = 100_000
EPS_START, EPS_END, EPS_STEPS = 1.0, 0.05, 10_000
ROLLOUT = 2048
GRID_POINTS = np.arange(1000, TOTAL_STEPS + 1, 1000)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_dqn(seed, double):
    set_seed(seed)
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                     double=double)
    state, _ = env.reset(seed=seed)
    ep_reward, records = 0.0, []

    for step in range(1, TOTAL_STEPS + 1):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_STEPS)
        action = agent.act(state, eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.buffer.add(state, action, reward, next_state, float(terminated))
        agent.update()
        state = next_state
        ep_reward += reward

        if terminated or truncated:
            records.append((step, ep_reward))
            ep_reward = 0.0
            state, _ = env.reset()

    env.close()
    return records


def run_ppo(seed):
    set_seed(seed)
    env = gym.make("CartPole-v1")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    state, _ = env.reset(seed=seed)
    ep_reward, steps, records = 0.0, 0, []

    while steps < TOTAL_STEPS:
        states, actions, logps, rewards, dones, values = [], [], [], [], [], []

        for _ in range(ROLLOUT):
            action, logp, value = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

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
                records.append((steps, ep_reward))
                ep_reward = 0.0
                state, _ = env.reset()

        agent.update(states, actions, logps, rewards, dones, values,
                     agent.value(state))

    env.close()
    return records


def to_curve(records, window=10):
    """Average reward of the last `window` finished episodes at each grid step."""
    steps = np.array([r[0] for r in records])
    rewards = np.array([r[1] for r in records])
    curve = []
    for g in GRID_POINTS:
        n = np.searchsorted(steps, g, side="right")
        curve.append(rewards[max(0, n - window):n].mean() if n > 0 else np.nan)
    return np.array(curve)


ALGOS = {
    "DQN": lambda seed: run_dqn(seed, double=False),
    "Double DQN": lambda seed: run_dqn(seed, double=True),
    "PPO": run_ppo,
}


def main():
    rows, curves = [], {name: [] for name in ALGOS}

    for name, run in ALGOS.items():
        for seed in SEEDS:
            print(f"Running {name}, seed {seed} ...", flush=True)
            curve = to_curve(run(seed))
            curves[name].append(curve)
            for step, value in zip(GRID_POINTS, curve):
                rows.append({"algo": name, "seed": seed,
                             "step": int(step), "reward": value})

    df = pd.DataFrame(rows)
    df.to_csv("results/algo_comparison.csv", index=False)

    plt.figure(figsize=(9, 5))
    for name, runs in curves.items():
        arr = np.vstack(runs)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        plt.plot(GRID_POINTS, mean, label=name)
        plt.fill_between(GRID_POINTS, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Environment steps")
    plt.ylabel("Reward (average of last 10 episodes)")
    plt.title(f"CartPole-v1: DQN vs Double DQN vs PPO ({len(SEEDS)} seeds)")
    plt.legend()
    plt.savefig("results/algo_comparison.png", dpi=150)

    final = df[df.step > 0.8 * TOTAL_STEPS].groupby(["algo", "seed"])["reward"].mean()
    print("\nAverage reward over the last 20% of training:")
    print(final.groupby("algo").agg(["mean", "std"]).round(1))


if __name__ == "__main__":
    main()
