import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        ids = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[ids], device=device),
            torch.as_tensor(self.actions[ids], device=device),
            torch.as_tensor(self.rewards[ids], device=device),
            torch.as_tensor(self.next_states[ids], device=device),
            torch.as_tensor(self.dones[ids], device=device),
        )

    def __len__(self):
        return self.size
