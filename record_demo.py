import gymnasium as gym
import imageio
import numpy as np
import torch

from common.networks import QNetwork
from algos.ppo import ActorCritic


def load_dqn():
    net = QNetwork(4, 2)
    net.load_state_dict(torch.load("results/dqn_cartpole.pt"))
    net.eval()

    def policy(state):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(net(s).argmax(dim=1).item())
    return policy


def load_ppo():
    model = ActorCritic(4, 2)
    model.load_state_dict(torch.load("results/ppo_cartpole.pt"))
    model.eval()

    def policy(state):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(model.actor(s).argmax(dim=1).item())   # most likely action
    return policy


def play(policy, seed, render=False):
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    state, _ = env.reset(seed=seed)
    total, frames, done = 0.0, [], False
    while not done:
        if render:
            frames.append(env.render())
        state, reward, terminated, truncated, _ = env.step(policy(state))
        total += reward
        done = terminated or truncated
    env.close()
    return total, frames


for name, loader in [("dqn", load_dqn), ("ppo", load_ppo)]:
    policy = loader()

    scores = [play(policy, seed=1000 + i)[0] for i in range(20)]
    print(f"{name.upper()}: mean {np.mean(scores):.1f} +/- {np.std(scores):.1f} "
          f"over 20 evaluation episodes")

    total, frames = play(policy, seed=123, render=True)
    imageio.mimsave(f"results/{name}_demo.gif", frames[::2], fps=30)
    print(f"  saved results/{name}_demo.gif (demo episode reward {total:.0f})")
