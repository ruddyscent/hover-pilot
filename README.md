# HoverPilot

![License](https://img.shields.io/badge/license-MIT-green)

Minimal Python client to connect to RealFlight Link (TCP 18083), exchange RC
commands, and expose a Gymnasium-compatible hover environment.

## Quickstart

Recommended with `uv`:

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
