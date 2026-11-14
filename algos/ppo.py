import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def dist_and_value(self, x):
        dist = Categorical(logits=self.actor(x))
        value = self.critic(x).squeeze(-1)
        return dist, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, batch_size=64, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.5, device="cpu"):
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.batch_size = epochs, batch_size
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    def _tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def act(self, state):
        with torch.no_grad():
            dist, value = self.model.dist_and_value(self._tensor(state).unsqueeze(0))
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def value(self, state):
        with torch.no_grad():
            _, value = self.model.dist_and_value(self._tensor(state).unsqueeze(0))
            return value.item()

    def update(self, states, actions, log_probs, rewards, dones, values, last_value):
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        n = len(rewards)

        # --- GAE: walk backwards through the rollout ---
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values          # targets for the critic

        states_t = self._tensor(states)
        actions_t = self._tensor(actions, torch.int64)
        old_logp = self._tensor(log_probs)
        adv_t = self._tensor(advantages)
        ret_t = self._tensor(returns)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # --- several epochs of minibatch updates on the same rollout ---
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                mb = perm[start:start + self.batch_size]

                dist, value = self.model.dist_and_value(states_t[mb])
                new_logp = dist.log_prob(actions_t[mb])
                entropy = dist.entropy().mean()

                ratio = (new_logp - old_logp[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_t[mb])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()
