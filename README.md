# HoverPilot

![License](https://img.shields.io/badge/license-MIT-green)

Minimal Python 3.10+ client to connect to RealFlight Link (TCP 18083), exchange
RC commands, and expose a Gymnasium-compatible hover environment.

## Installation

Install the base package from PyPI:

```bash
pip install hover-pilot
```

Install the optional PPO training, evaluation, and reporting dependencies:

```bash
pip install "hover-pilot[rl]"
```

With uv, use `uv add hover-pilot` or `uv add "hover-pilot[rl]"`.

## Quickstart

From a source checkout, install dependencies and run the demo with `uv`:

```bash
uv sync
cp .env.example .env
uv run hoverpilot-demo
```

Install the optional reinforcement-learning dependencies to train or play a PPO
policy:

```bash
uv sync --extra rl
uv run hoverpilot-ppo train --timesteps 50000 --save-path ppo_hoverpilot.pt
```

HoverPilot periodically evaluates the deterministic policy, keeps latest and
best checkpoints separately, and stores the complete training state for exact
continuation. A repository checkout also includes an example TOML experiment:

```bash
uv run hoverpilot-ppo train --config configs/elevator.toml
```

Evaluate the best checkpoint or generate an HTML summary from a TensorBoard run:

```bash
uv run hoverpilot-ppo evaluate --checkpoint ppo_hoverpilot.best.pt
uv run hoverpilot-ppo report runs/hoverpilot-ppo
```

The user guide describes TOML overrides, checkpoint comparison, full-state
resume, evaluation metrics, and HTML reports in detail.

By default HoverPilot connects to RealFlight Link at `127.0.0.1:18083`. Set
`RFLINK_HOST` when RealFlight runs on another host or outside the current network
namespace.

## Gymnasium Environment

```python
import numpy as np

from hoverpilot.config import HOST, PORT
from hoverpilot.envs import HoverPilotHoverEnv

env = HoverPilotHoverEnv(host=HOST, port=PORT, max_episode_steps=250)
observation, info = env.reset()
action = np.asarray([0.0, 0.0, 0.55, 0.0], dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(action)
```

The action contains `aileron`, `elevator`, `throttle`, and `rudder`. See the user
guide for observation layouts, reward and termination behavior, and trainer modes.

## Documentation

- [User guide](docs/user-guide.md): setup, running, training, validation, and troubleshooting
- [RealFlight Link interface](docs/realflight-link-interface.md): client API and protocol reference

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
