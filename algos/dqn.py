import copy
import random

import torch
import torch.nn.functional as F

from common.networks import QNetwork
from common.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 buffer_size=50_000, batch_size=64, target_update=500,
                 double=False, device="cpu"):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.double = double
        self.device = device

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, state_dim)
        self.learn_steps = 0

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.q_net(s).argmax(dim=1).item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size, self.device)
        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                # online net CHOOSES the best next action,
                # target net EVALUATES it
                best_a = self.q_net(s2).argmax(dim=1, keepdim=True)
                next_q = self.target_net(s2).gather(1, best_a).squeeze(1)
            else:
                # plain DQN: target net both chooses and evaluates
                next_q = self.target_net(s2).max(dim=1).values
            target = r + self.gamma * next_q * (1 - d)

        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
