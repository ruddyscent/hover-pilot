# HoverPilot

![License](https://img.shields.io/badge/license-MIT-green)

Minimal Python 3.10+ client to connect to RealFlight Link (TCP 18083), exchange
RC commands, and expose a Gymnasium-compatible hover environment.

## Installation

HoverPilot requires Python 3.10 or newer and a running copy of RealFlight with
RealFlight Link enabled. Install the base package from PyPI:

```bash
pip install hover-pilot
```

Install the optional PPO training, evaluation, and reporting dependencies:

```bash
pip install "hover-pilot[rl]"
```

With uv, use `uv tool install hover-pilot` for the command-line tools, or
`uv add hover-pilot` when adding the library to another project.

## Quickstart

### 1. Prepare RealFlight

Before running HoverPilot:

1. Start RealFlight and load an Airplane Hover Trainer scenario.
2. Enable RealFlight Link and confirm it listens on TCP port `18083`.
3. If RealFlight is on another computer, set `RFLINK_HOST` to an address that is
   reachable from the machine running HoverPilot.

```bash
export RFLINK_HOST=127.0.0.1  # replace when RealFlight runs elsewhere
export RFLINK_PORT=18083
```

### 2. Check connectivity safely

The doctor opens and closes a TCP connection only. It does not inject a
controller or send flight controls.

```bash
hoverpilot-doctor
```

Do not continue until this reports `OK`.

### 3. Validate the environment

The default validation uses neutral controls with zero throttle:

```bash
hoverpilot-validate --episodes 1 --max-episode-steps 50
```

Add `--control-test` only when you intentionally want to send random controls
to test action scaling.

### 4. Run the bounded demo

The demo sends neutral aileron, elevator, and rudder with throttle `0.55`. It
stops after 100 control steps by default:

```bash
hoverpilot-demo
```

Inspect options before changing its duration or throttle:

```bash
hoverpilot-demo --help
```

### Source checkout

Contributors working from a source checkout can install and run the same flow
with `uv`:

```bash
uv sync
cp .env.example .env
uv run hoverpilot-doctor
uv run hoverpilot-validate --episodes 1 --max-episode-steps 50
uv run hoverpilot-demo
```

## First PPO Training Run

Install the optional reinforcement-learning dependencies before training,
evaluation, or playback:

```bash
pip install "hover-pilot[rl]"
```

Create the maintained elevator starter configuration rather than configuring
PPO options individually:

```bash
hoverpilot-ppo init-config
hoverpilot-ppo train --config hoverpilot-elevator.toml
```

From a source checkout, use `uv sync --extra rl` and the checked-in copy:

HoverPilot periodically evaluates the deterministic policy, keeps latest and
best checkpoints separately, and stores the complete training state for exact
continuation. A repository checkout also includes an example TOML experiment:

```bash
uv run hoverpilot-ppo train --config configs/elevator.toml
```

Evaluate the best checkpoint or generate an HTML summary from a TensorBoard run:

```bash
uv run hoverpilot-ppo evaluate --checkpoint checkpoints/elevator.best.pt
uv run hoverpilot-ppo report runs/elevator
```

The user guide describes TOML overrides, checkpoint comparison, full-state
resume, evaluation metrics, and HTML reports in detail.

See the user guide for advanced overrides.

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
