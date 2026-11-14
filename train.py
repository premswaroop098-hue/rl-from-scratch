import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from algos.dqn import DQNAgent

SEED = 0
EPISODES = 400
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.985


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, device="cpu")

    records = []
    for episode in range(EPISODES):
        epsilon = max(EPS_END, EPS_START * EPS_DECAY ** episode)
        state, _ = env.reset(seed=SEED + episode)
        total_reward, losses, done = 0.0, [], False

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store only `terminated` as "done": hitting the 500-step time
            # limit doesn't mean the future is worthless
            agent.buffer.add(state, action, reward, next_state, float(terminated))

            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        records.append({
            "episode": episode,
            "reward": total_reward,
            "epsilon": epsilon,
            "loss": np.mean(losses) if losses else np.nan,
        })
        if episode % 20 == 0:
            print(f"Episode {episode:4d} | reward {total_reward:6.1f} | eps {epsilon:.2f}")

    df = pd.DataFrame(records)
    df["reward_ma"] = df["reward"].rolling(20).mean()
    df.to_csv("results/dqn_cartpole.csv", index=False)
    torch.save(agent.q_net.state_dict(), "results/dqn_cartpole.pt")

    plt.plot(df["episode"], df["reward"], alpha=0.3, label="episode reward")
    plt.plot(df["episode"], df["reward_ma"], label="20-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CartPole-v1")
    plt.legend()
    plt.savefig("results/dqn_cartpole.png", dpi=150)
    print("Saved results/dqn_cartpole.csv, .png and .pt")


if __name__ == "__main__":
    main()
