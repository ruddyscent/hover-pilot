from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError as exc:
    raise ImportError("PyTorch is required to use PPO training. Install it with `pip install torch`.") from exc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from hoverpilot.config import HOST, PORT
from hoverpilot.envs import HoverPilotHoverEnv
from hoverpilot.training.hover import RewardConfig
from hoverpilot.utils.logger import format_debug_state


WAITING_LOG_INTERVAL_S = 0.75
DEFAULT_WAIT_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_INITIAL_ACTION = np.asarray([0.0, 0.0, 0.55, 0.0], dtype=np.float32)


@dataclass
class PPOConfig:
    host: str = HOST
    port: int = PORT
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    max_episode_steps: Optional[int] = 300
    sleep_interval_s: float = 0.0

    timesteps: int = 50_000
    n_steps: int = 1024
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: Optional[int] = None
    save_path: str = "ppo_hoverpilot.pt"
    eval_episodes: int = 3
    log_interval: int = 1
    initial_action: Tuple[float, float, float, float] = (0.0, 0.0, 0.55, 0.0)
    wait_action: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    tensorboard_log_dir: Optional[str] = "runs/hoverpilot-ppo"
    device: str = "auto"


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super().__init__()
        hidden_sizes = [128, 128]
        layers = []
        input_size = observation_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU(inplace=True))
            input_size = hidden
        self.shared = nn.Sequential(*layers)
        self.policy_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        with torch.no_grad():
            self.policy_mean.bias.zero_()
            if action_dim >= 3:
                # Hover training needs non-zero throttle from the first step.
                self.policy_mean.bias[2] = float(DEFAULT_INITIAL_ACTION[2])
                self.policy_log_std[2] = -1.0

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(obs)
        mean = self.policy_mean(hidden)
        value = self.value_head(hidden).squeeze(-1)
        log_std = self.policy_log_std.expand_as(mean)
        return mean, log_std, value

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, value, mean


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
            delta = self.rewards[step] + gamma * next_value * self.dones[step] - self.values[step]
            gae = delta + gamma * lam * self.dones[step] * gae
            self.advantages[step] = gae
        self.returns[: self.index] = self.advantages[: self.index] + self.values[: self.index]

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


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested, but PyTorch cannot access an MPS device.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device {requested!r}; choose auto, cpu, cuda, or mps.")
    return torch.device(requested)


class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.env = self._build_env()
        observation_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        self.model = ActorCritic(observation_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.writer = self._build_writer()
        if config.seed is not None:
            self.seed(config.seed)

    def _build_writer(self):
        if self.config.tensorboard_log_dir is None:
            return None
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requires `tensorboard`. Install the RL extra with "
                "`uv sync --extra rl`."
            )
        return SummaryWriter(log_dir=self.config.tensorboard_log_dir)

    def _wait_action(self) -> np.ndarray:
        return np.asarray(self.config.wait_action, dtype=np.float32)

    def _initial_action(self) -> np.ndarray:
        return np.asarray(self.config.initial_action, dtype=np.float32)

    def _format_action_stats(self, actions: np.ndarray) -> str:
        labels = ("ail", "ele", "thr", "rud")
        parts = []
        for index, label in enumerate(labels[: actions.shape[1]]):
            column = actions[:, index]
            parts.append(f"{label}=mean:{column.mean():+.3f} std:{column.std():.3f}")
        return " ".join(parts)

    def _write_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def _write_action_metrics(self, actions: np.ndarray, step: int):
        labels = ("aileron", "elevator", "throttle", "rudder")
        for index, label in enumerate(labels[: actions.shape[1]]):
            column = actions[:, index]
            self._write_scalar(f"train/action/{label}_mean", float(column.mean()), step)
            self._write_scalar(f"train/action/{label}_std", float(column.std()), step)

    def _write_termination_metrics(self, termination_reasons: list[str], step: int):
        counts = Counter(termination_reasons)
        total = max(1, len(termination_reasons))
        for reason, count in counts.items():
            self._write_scalar(f"train/termination/{reason}", float(count), step)
            self._write_scalar(f"train/termination_rate/{reason}", float(count) / total, step)

    def _format_reward_breakdown(self, info: Optional[Dict]) -> str:
        if not info:
            return ""
        breakdown = info.get("reward_breakdown")
        if not breakdown:
            return ""
        return (
            " "
            f"reward_terms(pos={breakdown.get('position_reward', 0.0):+.3f} "
            f"att={breakdown.get('attitude_reward', 0.0):+.3f} "
            f"vel={breakdown.get('velocity_penalty', 0.0):+.3f} "
            f"rate={breakdown.get('angular_rate_penalty', 0.0):+.3f} "
            f"boundary={breakdown.get('boundary_penalty', 0.0):+.3f} "
            f"terminal={breakdown.get('terminal_penalty', 0.0):+.3f})"
        )

    def _log_episode_start(self, info: dict[str, object]):
        debug_state = info.get("debug_state") if isinstance(info, dict) else None
        print(
            f"[PPO] episode start reason={info.get('episode_start_reason')} "
            f"waiting={info.get('waiting_for_reset')}"
        )
        if debug_state:
            print(f"[PPO] start state {format_debug_state(debug_state)}")

    def _log_episode_end(
        self,
        *,
        episode_length: int,
        episode_reward: float,
        info: dict[str, object],
    ):
        debug_state = info.get("debug_state") if isinstance(info, dict) else None
        print(
            f"[PPO] episode end steps={episode_length} reward={episode_reward:.3f} "
            f"reason={info.get('termination_reason')}"
            f"{self._format_reward_breakdown(info)}"
        )
        if debug_state:
            print(f"[PPO] end state {format_debug_state(debug_state)}")

    def _log_rollout_summary(
        self,
        *,
        total_steps: int,
        rollout: RolloutBuffer,
        actions: list[np.ndarray],
        rewards: list[float],
        termination_reasons: list[str],
        elapsed_s: float,
    ):
        if not rewards:
            return
        action_array = np.asarray(actions, dtype=np.float32)
        termination_counts = Counter(termination_reasons)
        reason_summary = ", ".join(
            f"{reason}:{count}" for reason, count in sorted(termination_counts.items())
        ) or "none"
        print(
            f"[PPO] rollout steps={total_steps}/{self.config.timesteps} "
            f"samples={rollout.index} reward_mean={np.mean(rewards):+.3f} "
            f"reward_min={np.min(rewards):+.3f} reward_max={np.max(rewards):+.3f} "
            f"done_rate={sum(1 for reason in termination_reasons if reason != 'incomplete') / max(1, rollout.index):.3f} "
            f"elapsed={elapsed_s:.1f}s"
        )
        print(f"[PPO] rollout actions {self._format_action_stats(action_array)}")
        print(f"[PPO] rollout terminations {reason_summary}")

    def _log_update_summary(
        self,
        *,
        total_steps: int,
        policy_losses: list[float],
        value_losses: list[float],
        entropy_values: list[float],
        ratio_values: list[float],
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ):
        print(
            f"[PPO] update steps={total_steps}/{self.config.timesteps} "
            f"policy_loss={np.mean(policy_losses):+.4f} "
            f"value_loss={np.mean(value_losses):+.4f} "
            f"entropy={np.mean(entropy_values):.4f} "
            f"ratio={np.mean(ratio_values):.4f} "
            f"return_mean={returns.mean().item():+.3f} "
            f"adv_mean={advantages.mean().item():+.3f} adv_std={advantages.std(unbiased=False).item():.3f}"
        )

    def _build_env(self) -> gym.Env:
        env = HoverPilotHoverEnv(
            host=self.config.host,
            port=self.config.port,
            reward_config=self.config.reward_config,
            max_episode_steps=self.config.max_episode_steps,
            sleep_interval_s=self.config.sleep_interval_s,
        )
        return env

    def seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)

    def _normalize_action(self, raw_action: np.ndarray) -> np.ndarray:
        low = self.env.action_space.low
        high = self.env.action_space.high
        return np.clip(raw_action, low, high)

    def train(self):
        total_steps = 0
        report_every = max(1, self.config.log_interval)
        training_start = time.time()
        episode_rewards = []
        episode_lengths = []
        if self.writer is not None:
            self.writer.add_text(
                "run/config",
                "\n".join(
                    [
                        f"timesteps={self.config.timesteps}",
                        f"n_steps={self.config.n_steps}",
                        f"batch_size={self.config.batch_size}",
                        f"epochs={self.config.epochs}",
                        f"learning_rate={self.config.learning_rate}",
                        f"max_episode_steps={self.config.max_episode_steps}",
                        f"seed={self.config.seed}",
                    ]
                ),
                0,
            )
        try:
            observation, info = reset_env_with_wait(
                self.env,
                action=self._wait_action(),
                initial_action=self._initial_action(),
            )
            self._log_episode_start(info)

            while total_steps < self.config.timesteps:
                rollout = RolloutBuffer(self.config.n_steps, *self.env.observation_space.shape, self.env.action_space.shape[0], self.device)
                episode_reward = 0.0
                episode_length = 0
                rollout_actions: list[np.ndarray] = []
                rollout_rewards: list[float] = []
                rollout_termination_reasons: list[str] = []
                for step in range(self.config.n_steps):
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_tensor, log_prob_tensor, value_tensor = self.model.get_action(obs_tensor)
                    action = action_tensor.squeeze(0).detach().cpu().numpy()
                    clipped_action = self._normalize_action(action)
                    next_obs, reward, terminated, truncated, info = self.env.step(clipped_action)
                    done = bool(terminated or truncated)
                    rollout_actions.append(clipped_action.copy())
                    rollout_rewards.append(float(reward))
                    rollout_termination_reasons.append(info.get("termination_reason") or ("truncated" if truncated else "incomplete"))
                    rollout.add(
                        observation,
                        clipped_action,
                        reward,
                        done,
                        float(value_tensor.item()),
                        float(log_prob_tensor.item()),
                    )
                    episode_reward += reward
                    episode_length += 1
                    observation = next_obs
                    total_steps += 1
                    if done:
                        self._log_episode_end(
                            episode_length=episode_length,
                            episode_reward=episode_reward,
                            info=info,
                        )
                        self._write_scalar("train/episode_reward", float(episode_reward), total_steps)
                        self._write_scalar("train/episode_length", float(episode_length), total_steps)
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
                        observation, info = reset_env_with_wait(
                            self.env,
                            action=self._wait_action(),
                            initial_action=self._initial_action(),
                        )
                        self._log_episode_start(info)
                        episode_reward = 0.0
                        episode_length = 0
                    if total_steps >= self.config.timesteps:
                        break

                last_value = self.model(torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))[2].item()
                rollout.compute_returns_and_advantages(last_value, self.config.gamma, self.config.gae_lambda)
                advantages = rollout.advantages[: rollout.index]
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
                returns = rollout.returns[: rollout.index]

                self._log_rollout_summary(
                    total_steps=total_steps,
                    rollout=rollout,
                    actions=rollout_actions,
                    rewards=rollout_rewards,
                    termination_reasons=rollout_termination_reasons,
                    elapsed_s=time.time() - training_start,
                )

                action_array = np.asarray(rollout_actions, dtype=np.float32)
                self._write_scalar("train/reward_mean", float(np.mean(rollout_rewards)), total_steps)
                self._write_scalar("train/reward_min", float(np.min(rollout_rewards)), total_steps)
                self._write_scalar("train/reward_max", float(np.max(rollout_rewards)), total_steps)
                self._write_scalar(
                    "train/done_rate",
                    float(sum(1 for reason in rollout_termination_reasons if reason != "incomplete") / max(1, rollout.index)),
                    total_steps,
                )
                self._write_scalar("train/return_mean", float(returns.mean().item()), total_steps)
                self._write_scalar("train/return_std", float(returns.std(unbiased=False).item()), total_steps)
                self._write_scalar("train/advantage_mean", float(advantages.mean().item()), total_steps)
                self._write_scalar("train/advantage_std", float(advantages.std(unbiased=False).item()), total_steps)
                self._write_action_metrics(action_array, total_steps)
                self._write_termination_metrics(rollout_termination_reasons, total_steps)

                policy_losses = []
                value_losses = []
                entropy_values = []
                ratio_values = []
                for epoch in range(self.config.epochs):
                    for batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in rollout.get_batches(self.config.batch_size):
                        batch_log_probs, batch_entropy, batch_values, _ = self.model.evaluate_actions(batch_obs, batch_actions)
                        ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                        surrogate1 = ratio * batch_advantages
                        surrogate2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * batch_advantages
                        policy_loss = -torch.min(surrogate1, surrogate2).mean()
                        value_loss = self.config.value_coef * (batch_returns - batch_values).pow(2).mean()
                        entropy_loss = -self.config.entropy_coef * batch_entropy.mean()
                        loss = policy_loss + value_loss + entropy_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        policy_losses.append(float(policy_loss.item()))
                        value_losses.append(float(value_loss.item()))
                        entropy_values.append(float(batch_entropy.mean().item()))
                        ratio_values.append(float(ratio.mean().item()))

                self._log_update_summary(
                    total_steps=total_steps,
                    policy_losses=policy_losses,
                    value_losses=value_losses,
                    entropy_values=entropy_values,
                    ratio_values=ratio_values,
                    returns=returns,
                    advantages=advantages,
                )
                self._write_scalar("train/policy_loss", float(np.mean(policy_losses)), total_steps)
                self._write_scalar("train/value_loss", float(np.mean(value_losses)), total_steps)
                self._write_scalar("train/entropy", float(np.mean(entropy_values)), total_steps)
                self._write_scalar("train/ratio", float(np.mean(ratio_values)), total_steps)

                if len(episode_rewards) >= report_every:
                    avg_reward = float(np.mean(episode_rewards[-report_every:]))
                    avg_length = float(np.mean(episode_lengths[-report_every:]))
                    elapsed = time.time() - training_start
                    print(
                        f"[PPO] steps={total_steps}/{self.config.timesteps} "
                        f"avg_reward={avg_reward:.3f} avg_length={avg_length:.1f} elapsed={elapsed:.1f}s"
                    )
                    self._write_scalar("train/avg_reward", avg_reward, total_steps)
                    self._write_scalar("train/avg_length", avg_length, total_steps)

                if self.writer is not None:
                    self.writer.flush()

            torch.save(self.model.state_dict(), self.config.save_path)
            print(f"Saved trained policy to {self.config.save_path}")
            self._evaluate_policy()
        finally:
            if self.writer is not None:
                self.writer.close()

    def _evaluate_policy(self):
        rewards = []
        lengths = []
        for episode in range(self.config.eval_episodes):
            observation, _ = reset_env_with_wait(
                self.env,
                action=self._wait_action(),
                initial_action=self._initial_action(),
            )
            episode_reward = 0.0
            episode_length = 0
            while True:
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    mean, _, _ = self.model(obs_tensor)
                action = mean.squeeze(0).cpu().numpy()
                clipped_action = self._normalize_action(action)
                observation, reward, terminated, truncated, info = self.env.step(clipped_action)
                episode_reward += reward
                episode_length += 1
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
            lengths.append(episode_length)
        avg_reward = float(np.mean(rewards))
        avg_length = float(np.mean(lengths))
        print(f"Evaluation: avg_reward={avg_reward:.3f}, avg_length={avg_length:.1f}")
        self._write_scalar("eval/avg_reward", avg_reward, self.config.timesteps)
        self._write_scalar("eval/avg_length", avg_length, self.config.timesteps)


def reset_env_with_wait(
    env: gym.Env,
    *,
    action: Optional[Union[np.ndarray, list, tuple]] = None,
    initial_action: Optional[Union[np.ndarray, list, tuple]] = None,
):
    if getattr(env, "_waiting_for_reset", False):
        poll_wait = getattr(env, "poll_wait_for_next_episode", None)
        if not callable(poll_wait):
            raise RuntimeError("environment reports waiting-for-reset but does not expose poll_wait_for_next_episode()")
        return _wait_for_episode_start(env, poll_wait=poll_wait, action=action)

    try:
        reset_options = None
        if initial_action is not None:
            reset_options = {"initial_action": initial_action}
        return env.reset(options=reset_options)
    except TimeoutError as exc:
        poll_wait = getattr(env, "poll_wait_for_next_episode", None)
        if not callable(poll_wait):
            raise

        print(f"waiting for trainer reset before episode | {exc}")
        return _wait_for_episode_start(env, poll_wait=poll_wait, action=action)


def _wait_for_episode_start(
    env: gym.Env,
    *,
    poll_wait,
    action: Optional[Union[np.ndarray, list, tuple]],
):
    del env
    last_wait_log_at = 0.0
    while True:
        started, observation, info = poll_wait(action=action)
        if started:
            return observation, info
        now = time.monotonic()
        if now - last_wait_log_at >= WAITING_LOG_INTERVAL_S:
            print(f"waiting for trainer reset | {format_debug_state(info.get('debug_state'))}")
            last_wait_log_at = now


def validate_environment(host: str, port: int, episodes: int = 2, max_episode_steps: Optional[int] = 100):
    env = HoverPilotHoverEnv(
        host=host,
        port=port,
        max_episode_steps=max_episode_steps,
    )
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Sample observation:", env.observation_space.sample())

    for episode in range(episodes):
        observation, info = reset_env_with_wait(env)
        print(f"Episode {episode + 1} reset observation shape={observation.shape}, reason={info.get('episode_start_reason')}")
        if info.get("episode_readiness"):
            print("  readiness:", info["episode_readiness"])
        step = 0
        while step < max_episode_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"  step={step} reward={reward:.3f} terminated={terminated} truncated={truncated} "
                f"reason={info.get('termination_reason')}"
            )
            if terminated or truncated:
                break
            step += 1
        if step == max_episode_steps:
            print("  reached max_episode_steps without termination")
    env.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on HoverPilot Hover Env or validate environment quality.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    train_parser = subparsers.add_parser("train", help="Train a PPO policy")
    train_parser.add_argument("--timesteps", type=int, default=50_000)
    train_parser.add_argument("--save-path", type=str, default="ppo_hoverpilot.pt")
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--max-episode-steps", type=int, default=300)
    train_parser.add_argument("--sleep-interval-s", type=float, default=0.0)
    train_parser.add_argument("--n-steps", type=int, default=1024)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--clip-epsilon", type=float, default=0.2)
    train_parser.add_argument("--value-coef", type=float, default=0.5)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--max-grad-norm", type=float, default=0.5)
    train_parser.add_argument("--log-interval", type=int, default=1)
    train_parser.add_argument("--eval-episodes", type=int, default=3)
    train_parser.add_argument("--tensorboard-log-dir", type=str, default="runs/hoverpilot-ppo")
    train_parser.add_argument("--disable-tensorboard", action="store_true")
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Training device. auto selects CUDA when available and otherwise CPU; MPS is opt-in.",
    )
    train_parser.add_argument("--host", type=str, default=HOST)
    train_parser.add_argument("--port", type=int, default=PORT)

    validate_parser = subparsers.add_parser("validate", help="Validate environment reset, observation, reward, and action scaling")
    validate_parser.add_argument("--episodes", type=int, default=2)
    validate_parser.add_argument("--max-episode-steps", type=int, default=100)
    validate_parser.add_argument("--host", type=str, default=HOST)
    validate_parser.add_argument("--port", type=int, default=PORT)

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.command == "train":
        config = PPOConfig(
            host=args.host,
            port=args.port,
            timesteps=args.timesteps,
            max_episode_steps=args.max_episode_steps,
            sleep_interval_s=args.sleep_interval_s,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            save_path=args.save_path,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            tensorboard_log_dir=None if args.disable_tensorboard else args.tensorboard_log_dir,
            device=args.device,
        )
        trainer = PPOTrainer(config)
        trainer.train()
    elif args.command == "validate":
        validate_environment(args.host, args.port, episodes=args.episodes, max_episode_steps=args.max_episode_steps)


if __name__ == "__main__":
    main()
