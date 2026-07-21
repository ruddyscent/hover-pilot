# HoverPilot

![License](https://img.shields.io/badge/license-MIT-green)

Minimal Python client to connect to RealFlight Link (TCP 18083), exchange RC commands, and expose a Gymnasium-compatible hover environment.

## Quickstart

Recommended with `uv`:

```bash
uv sync
cp .env.example .env
uv run hoverpilot-demo
```

If you prefer module execution, this also works after `uv sync`:

```bash
uv run python -m hoverpilot.main
```

Legacy `pip` workflow:

```bash
python3 -m venv .venv
. .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
python -m hoverpilot.main
```

## Gymnasium Environment

The project now exposes a Gymnasium-style environment:

```python
import numpy as np

from hoverpilot.config import HOST, PORT
from hoverpilot.envs import HoverPilotHoverEnv

env = HoverPilotHoverEnv(
    host=HOST,
    port=PORT,
    max_episode_steps=250,
)
observation, info = env.reset()
action = np.asarray([0.0, 0.0, 0.55, 0.0], dtype=np.float32)
observation, reward, terminated, truncated, info = env.step(action)
```

The API mirrors Gymnasium:

- `reset(...) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`

Action format is a 4-element `float32` array:

- index 0: `aileron` in `[-1, 1]`
- index 1: `elevator` in `[-1, 1]`
- index 2: `throttle` in `[0, 1]`
- index 3: `rudder` in `[-1, 1]`

Observation format is a compact 12-element `float32` vector for hover training:

- position: `x`, `y`, `altitude_agl`
- attitude: `roll`, `inclination`, `azimuth`
- world velocity: `u`, `v`, `w`
- angular rates: `pitch_rate`, `roll_rate`, `yaw_rate`

Reward and termination are integrated from `hoverpilot.training.hover`:

- reward prefers staying near the target hover point and upright attitude
- boundary proximity adds a growing penalty before failure
- terminal failures include trainer boundary exit, altitude bounds, lost components, locked vehicle states, configured controller inactivity, configured engine stop, and post-start ground contact

`reset()` waits for a usable start state before returning. During this warmup the environment keeps sending a safe idle action and polls RealFlight until readiness is satisfied or a timeout is reached.

### RealFlight Link host

By default the client connects to `127.0.0.1:18083`. That is only correct when
Python and RealFlight are running in the same network namespace. From a Docker
container, VM, WSL instance, or Jetson talking to another machine, set
`RFLINK_HOST` to the host/IP address that is reachable from that shell:

```bash
RFLINK_HOST=<realflight-host-ip> uv run --no-sync hoverpilot-demo
```

If startup reports `unable to connect to RealFlight Link`, verify that
RealFlight is running, RealFlight Link is enabled, TCP port `18083` is reachable,
and `RFLINK_HOST` is not still pointing at the container's own loopback address.

## Demo

Run:

```bash
python -m hoverpilot.main
```

The demo prints:

- observation shape
- scalar reward
- `terminated` / `truncated`
- termination reason when present
- current AGL altitude from `info["debug_state"]`
- a concise RealFlight state summary

The demo keeps running across episodes and only stops on `KeyboardInterrupt`. During reset wait periods it rate-limits the `waiting for trainer reset` log to avoid flooding the terminal.

## Episode Lifecycle

`Airplane Hover Trainer` does not always expose a clean explicit reset flag through RealFlight Link, so the environment manages episode lifecycle conservatively.

Episode start:

- `reset()` and reset-wait polling both use a safe idle action.
- A state is considered ready when it is not obviously uninitialized, not locked, and not already failed.
- Controller-active and engine-running checks are available as configurable readiness gates because these fields can behave differently across trainer modes.
- Ground contact is allowed during startup by default because some trainer resets spawn on or very near the ground.

Episode end:

- Hard terminal failures include:
  - `m_hasLostComponents > threshold`
  - boundary exit in `x` or `y`
  - altitude too low / too high
  - locked vehicle state
  - configured controller inactive / engine stopped conditions
  - post-start ground contact after the configured grace period
- `m_currentAircraftStatus` is currently treated as opaque. It is exposed in `info["debug_state"]`, and only becomes terminal if you explicitly configure known terminal status codes.

Reset-wait and restart:

- After termination, the environment keeps polling with a safe idle action until a new episode can be started.
- Restart signals are checked in this order:
  - reset button pressed
  - physics time rollback
  - trainer-driven reposition / teleport into a reset-like stationary state
- In the current Airplane Hover Trainer setup, the more semantic crash / recovery flags
  (`m_hasLostComponents`, `m_anEngineIsRunning`, `m_isTouchingGround`) often stay fixed at
  `0`, so they are still exposed in `debug_state` but are not used as primary reset signals.
  You can verify that behavior with:
  `RFLINK_DEBUG_STATE_FLAGS=1 python -m hoverpilot.main`

Useful tuning parameters on `HoverPilotHoverEnv`:

- readiness / warmup:
  - `max_reset_wait_seconds`
  - `reset_poll_interval_seconds`
  - `ready_controller_active_threshold`
  - `ready_running_threshold`
  - `ready_locked_threshold`
  - `allow_ground_contact_at_ready`
- teleport fallback:
  - `reposition_speed_threshold_mps`
  - `reset_teleport_distance_m`
- termination thresholds via `RewardConfig`:
  - `controller_active_threshold`
  - `terminate_on_engine_stopped`
  - `ground_contact_grace_seconds`
  - `known_terminal_aircraft_status_codes`

The environment prefers explicit RealFlight Link reset signals first. Teleport / reposition detection is kept as a fallback because the Hover Trainer can reset by suddenly moving the aircraft without updating the more semantic lifecycle flags.

## PPO Training and Environment Validation

A lightweight PPO trainer is now available in `hoverpilot.rl.ppo`.

### macOS Metal / MPS

The shared Dev Container is intended for reproducible CPU development and testing.
Docker Desktop runs it inside a Linux virtual machine and does not expose the macOS
Metal Performance Shaders (MPS) device to the container. To use PyTorch with Metal
acceleration on an Apple Silicon Mac, install and run HoverPilot directly on the
macOS host instead of inside the Dev Container:

```bash
uv sync --extra rl
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

The availability check must print `True` before an application can use MPS. MPS is
deliberately opt-in because this project's small policy network and sequential
simulator communication may be faster on CPU. To try MPS, select it explicitly:

```bash
uv run hoverpilot-ppo train --device mps --timesteps 50000 --save-path ppo_hoverpilot.pt
```

The `--device` option accepts `auto`, `cpu`, `cuda`, or `mps`. The default `auto`
mode selects CUDA when available and otherwise uses CPU; it never selects MPS
implicitly. The Dev Container remains useful for CPU development and tests, while
the Jetson GPU workflow is handled separately by `compose.jetson.yml`.

## NVIDIA Jetson with NGC PyTorch Container

HoverPilot can be run on NVIDIA Jetson inside an NVIDIA NGC PyTorch container using
the provided Compose file [compose.jetson.yml](/Users/kwchun/Workspace/hover-pilot/compose.jetson.yml).

Prerequisites on the Jetson host:

- JetPack 5.0.2 or newer installed on the device
- NVIDIA Container Toolkit configured for Docker
- Docker access for your user
- NGC login completed with `docker login nvcr.io`

HoverPilot's Jetson container workflow assumes a JetPack 5.x class environment with
Ubuntu 20.04 / Python 3.8. JetPack 5.0.2 is the minimum supported baseline in this
README because it is the first production-quality JetPack 5 release and supports
NVIDIA Jetson Xavier NX modules.

The exact NGC image tag must match the JetPack / L4T release on the device.
Use [.env.example](/Users/kwchun/Workspace/hover-pilot/.env.example) as a template:

```bash
cp .env.example .env
```

Then set the NGC image in `.env`:

```bash
HOVERPILOT_NGC_IMAGE=nvcr.io/nvidia/<official-jetson-pytorch-image>:<matching-tag>
```

Bring up the container:

```bash
docker compose -f compose.jetson.yml up -d
docker compose -f compose.jetson.yml exec hoverpilot bash
```

The project source tree is mounted into `/workspace/hover-pilot` inside the container.

### Using uv Without Replacing Container PyTorch

The NGC image already includes NVIDIA's Jetson-optimized PyTorch build.
To keep `uv` from replacing that PyTorch installation:

```bash
cd /workspace/hover-pilot
export UV_PYTHON=/usr/bin/python3
uv venv --python /usr/bin/python3 --system-site-packages
source .venv/bin/activate
python3 -c "import torch; print(torch.__version__)"
uv sync --python /usr/bin/python3 --extra rl --no-install-package torch --inexact
python3 -c "import torch; print(torch.__version__)"
```

Why this setup:

- `uv venv --python /usr/bin/python3 --system-site-packages` creates the project environment from the Jetson container's system Python instead of a uv-managed Python
- `--system-site-packages` lets the virtual environment see the PyTorch already installed in the container
- `--no-install-package torch` tells `uv` not to install its own `torch`
- `--inexact` avoids removing packages already provided by the container image

If `uv sync` prints a different interpreter version and recreates `.venv`, remove the
environment and repeat the steps above with `UV_PYTHON=/usr/bin/python3` set.

After that, prefer `uv run --no-sync ...` so execution does not try to resync and replace packages:

```bash
uv run --no-sync hoverpilot-demo
uv run --no-sync hoverpilot-validate --episodes 2 --max-episode-steps 100
uv run --no-sync hoverpilot-ppo train --timesteps 50000 --save-path ppo_hoverpilot.pt
```

Run TensorBoard from inside the container:

```bash
uv run --no-sync tensorboard --host 0.0.0.0 --port 6006 --logdir runs
```

Because the Compose file uses host networking on Jetson, TensorBoard is then available at:

```text
http://<jetson-ip>:6006
```

Install the optional RL dependency:

```bash
uv sync --extra rl
```

If torch installation fails (e.g., on Alpine aarch64), install the base package instead:

```bash
uv sync
```

Train a policy (requires torch):

```bash
uv run hoverpilot-ppo train --timesteps 50000 --save-path ppo_hoverpilot.pt
```

Training writes TensorBoard logs to `runs/hoverpilot-ppo` by default.

Customize training with additional options:

```bash
uv run hoverpilot-ppo train --timesteps 50000 \
  --save-path ppo_hoverpilot.pt \
  --max-episode-steps 300 \
  --n-steps 2048 \
  --batch-size 128 \
  --epochs 10 \
  --learning-rate 3e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-epsilon 0.2 \
  --value-coef 0.5 \
  --entropy-coef 0.01 \
  --max-grad-norm 0.5 \
  --eval-episodes 5 \
  --log-interval 10 \
  --tensorboard-log-dir runs/hoverpilot-ppo-exp1 \
  --seed 42
```

The PPO CLI now exposes these tuning parameters directly:

- `--n-steps`
- `--batch-size`
- `--epochs`
- `--learning-rate`
- `--gamma`
- `--gae-lambda`
- `--clip-epsilon`
- `--value-coef`
- `--entropy-coef`
- `--max-grad-norm`

To disable TensorBoard logging for a run:

```bash
uv run hoverpilot-ppo train --timesteps 50000 --disable-tensorboard
```

Monitor training with TensorBoard:

```bash
uv run tensorboard --logdir runs
```

Then open `http://localhost:6006`.

Useful TensorBoard scalars include:

- `train/episode_reward`
- `train/episode_length`
- `train/reward_mean`
- `train/action/throttle_mean`
- `train/termination/parked_on_ground`
- `train/policy_loss`
- `train/value_loss`
- `train/entropy`
- `eval/avg_reward`

Validate the environment before training:

```bash
uv run hoverpilot-validate --episodes 2 --max-episode-steps 100
```

This validation command helps confirm:

- `reset()` behavior and `episode_start_reason`
- observation shape and bounds
- action scaling across the drone control channels
- reward and termination signals during short episodes
- whether boundary termination is firing too aggressively

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
