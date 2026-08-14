"""On-device rollout storage and PPO minibatch generation."""

from __future__ import annotations

import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, device: torch.device):
        self.device = device
        self.observations = torch.zeros((capacity, observation_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.index = 0
        self.capacity = capacity

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        if self.index >= self.capacity:
            raise IndexError("RolloutBuffer is full")
        self.observations[self.index].copy_(torch.as_tensor(observation, dtype=torch.float32, device=self.device))
        self.actions[self.index].copy_(torch.as_tensor(action, dtype=torch.float32, device=self.device))
        self.rewards[self.index] = reward
        self.dones[self.index] = 0.0 if done else 1.0
        self.values[self.index] = value
        self.log_probs[self.index] = log_prob
        self.index += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        lam: float,
    ):
        gae = 0.0
        last_value_tensor = torch.tensor(last_value, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.index)):
            next_value = last_value_tensor if step == self.index - 1 else self.values[step + 1]
            # The one-step TD error follows Eq. (6.5)
            # (Sutton & Barto, 2018, Section 6.1).
            # The backward weighted accumulation is conceptually related to
            # the lambda-return; GAE applies that idea to advantage estimates
            # (Sutton & Barto, 2018, Sections 12.1-12.2).
            # http://incompleteideas.net/book/the-book-2nd.html
            delta = self.rewards[step] + gamma * next_value * self.dones[step] - self.values[step]
            gae = delta + gamma * lam * self.dones[step] * gae
            self.advantages[step] = gae
        self.returns[: self.index] = self.advantages[: self.index] + self.values[: self.index]

    def normalize_advantages(self) -> torch.Tensor:
        advantages = self.advantages[: self.index]
        normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        advantages.copy_(normalized)
        return advantages

    def get_batches(self, batch_size: int):
        indices = torch.randperm(self.index, device=self.device)
        for start in range(0, self.index, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.observations[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx],
            )
