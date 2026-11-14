import sys
sys.path.append(".")

import torch
from common.networks import QNetwork
from common.replay_buffer import ReplayBuffer

net = QNetwork(state_dim=4, action_dim=2)
fake_state = torch.rand(1, 4)
print("Q-values for a random state:", net(fake_state))

buf = ReplayBuffer(capacity=100, state_dim=4)
for i in range(10):
    buf.add([0.1, 0.2, 0.3, 0.4], 1, 1.0, [0.2, 0.3, 0.4, 0.5], 0.0)
print("Buffer size:", len(buf))

s, a, r, s2, d = buf.sample(batch_size=5)
print("Sampled states shape:", s.shape)
