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

Observation format is a normalized 13-element `float32` vector for hover training:

- target-relative position: scaled `x_error`, `y_error`, `altitude_error`
- attitude: scaled `roll`, scaled `inclination`, `sin(azimuth)`, `cos(azimuth)`
- scaled world velocity: `u`, `v`, `w`
- scaled angular rates: `pitch_rate`, `roll_rate`, `yaw_rate`

Elevator-only PPO uses a compact 6-element longitudinal observation instead:

- inclination tracking error relative to the position/velocity recovery target;
  nose-up hover at the origin is `0°`
- pitch rate
- horizontal position error projected onto the reset heading
- horizontal velocity projected onto the reset heading
- altitude error
- vertical world velocity

Observations are clipped to `[-5, 5]`. The PPO policy uses a tanh-squashed
Gaussian distribution so sampled actions stay inside the RealFlight channel
bounds without breaking the PPO log-probability ratio. Throttle is mapped to
`[0, 1]`; the other three channels are mapped to `[-1, 1]`.

Reward and termination are integrated from `hoverpilot.training.hover`:

- reward prefers staying near the target hover point and upright attitude
- boundary proximity adds a growing penalty before failure
- terminal failures include trainer cylinder exit, minimum-altitude failure, lost components, locked vehicle states, configured controller inactivity, configured engine stop, and post-start ground contact

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
  - exit from the trainer cylinder
  - altitude too low
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

For a RealFlight Hover Trainer configuration where PPO controls only elevator,
use a one-dimensional policy and keep the other transmitted channels fixed:

First verify that positive and negative elevator commands create opposite
pitch-rate responses. Reset Airplane Hover Trainer immediately before running:

```bash
uv run hoverpilot-ppo diagnose-elevator \
  --elevator-fixed-throttle 0.55 \
  --pulse 0.1
```

The diagnostic limits the pulse to `0.25`, records the neutral state before
each direction, and returns elevator to neutral on completion or failure. It
warns when the two pulses start from materially different pitch-rate or
inclination conditions, since that makes the polarity comparison unreliable.
Reset and repeat the diagnostic before training if either that warning appears
or the responses have the same sign.

```bash
uv run hoverpilot-ppo train --control-mode elevator \
  --elevator-fixed-throttle 0.55 \
  --rflink-socket-timeout-s 3 \
  --rflink-request-attempts 4 \
  --checkpoint-interval-steps 1024 \
  --save-path ppo_hoverpilot_elevator.pt \
  --tensorboard-log-dir runs/hoverpilot-ppo-elevator \
  --seed 42
```

RealFlight reports raw `m_inclination=90°` for the vertical, nose-up hover
attitude. Elevator mode converts that to a signed control error of `0°`.
Because azimuth flips by 180° at this Euler singularity, the signed error is
computed as `(inclination - 90°) * cos(azimuth - reset heading)`. This lets the
policy distinguish equal tilts on opposite sides of vertical. Position,
altitude, and heading are anchored at each trainer reset. Position and
longitudinal position rate command a signed target inclination, limited to
`30°`; at the origin the target remains the vertical `0°` error. The position
rate is calculated directly from consecutive positions and RealFlight physics
time rather than relying on the simulator U/V velocity polarity, which can
flip between vertical-hover resets. The target uses
`-(2 × position + 3 × position_rate)`: outward motion strengthens recovery
while inward motion starts braking before crossing the origin. This
state-dependent target is active from the beginning: small errors request a
small tilt, while large outward drift can immediately request the full `30°`.
The policy observes inclination relative to that recovery target rather than
raw vertical error.

The PPO-only elevator actor has two learned positive gains for attitude and
pitch rate. Mirroring inclination, pitch rate, position, and longitudinal
velocity therefore mirrors the elevator command, and the restoring sign cannot
reverse. The critic retains its nonlinear network, but the actor has no
separate MLP or direct position term that could cancel the recovery target.
Unlike `elevator-pd`, it injects no fixed prior or bounded residual.

Elevator training defaults to 300,000 steps, a `1e-4` learning rate, `0.08`
initial policy standard deviation, `0.0001` entropy coefficient, and 10 final
evaluation episodes. These settings use conservative exploration while keeping
the rollout on-policy. `--policy-preset none` is the default; the optional
`--policy-preset elevator-pd` supplies measured-sign restoring control while
PPO learns a bounded residual. Compare both presets when the goal is to
distinguish learned PPO performance from controller assistance.

Each trainer reposition defines a new local three-dimensional origin. The reset
position is `(x=0, y=0, z=0)`, where `z` is altitude AGL relative to the reset
altitude. The trainer boundary is modeled as a vertical cylinder with a 6 m
horizontal radius around the reset x/y position. Altitude does not reduce the
available horizontal radius, and no unverified upper ceiling is imposed. A new
process also requires the aircraft to be within 0.5° of the vertical attitude
before accepting a state as an episode start, so an in-progress fall is not
re-anchored as a new origin.

Continue training a structured elevator checkpoint with:

```bash
uv run hoverpilot-ppo train --control-mode elevator \
  --resume-from ppo_hoverpilot_elevator.pt \
  --save-path ppo_hoverpilot_elevator_continued.pt \
  --timesteps 10000
```

HoverPilot PPO checkpoints use structured format v2. They record policy/value
weights, control mode, policy preset, fixed elevator throttle, and the
observation/recovery scaling configuration. Version 1 checkpoints are rejected
because their observation and actor definitions are incompatible.
When resuming, `--control-mode` must match the checkpoint; the saved policy
preset and fixed throttle take precedence over their command-line defaults.
The optimizer is recreated with the command's current learning rate and tuning
options.

Training writes TensorBoard logs to `runs/hoverpilot-ppo` by default.

List all training options with:

```bash
uv run hoverpilot-ppo train --help
```

The task-specific switches are `--control-mode`, `--policy-preset`,
`--elevator-fixed-throttle`, `--resume-from`, and
`--checkpoint-interval-steps`.

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
- `train/optimization_reward_mean`
- `train/action/throttle_mean`
- `train/termination/parked_on_ground`
- `train/policy_loss`
- `train/value_loss`
- `train/entropy`
- `train/ratio`, `train/ratio_min`, `train/ratio_max`
- `train/approx_kl`
- `train/clip_fraction`
- `train/update_epochs`, `train/kl_early_stop`
- `train/value_mean`, `train/value_std`, `train/explained_variance`
- `train/action/<channel>_saturation_fraction`
- `train/control/elevator_action`
- `train/state/inclination_error_deg`, `train/state/abs_inclination_error_deg`
- `train/state/target_inclination_error_deg`
- `train/state/inclination_tracking_error_deg`
- `train/state/pitch_rate_deg_s`, `train/state/abs_pitch_rate_deg_s`
- `train/state/longitudinal_error_m`, `train/state/longitudinal_velocity_mps`
- `train/state/radial_distance_m`
- `train/recovery_probe/*`,
  `train/recovery_probe/minimum_restoring_margin`,
  `train/recovery_probe/effective_restoring_fraction`
- `eval/avg_reward`
- `eval/reward_per_step`
- `eval/position_error_m`, `eval/altitude_error_m`, `eval/attitude_error_deg`
- `eval/termination/*`, `eval/termination_rate/*`

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

For an elevator-only curriculum, begin with the smallest Airplane Hover Trainer
disturbance setting. Increase the trainer's initial angular/position
perturbations only after deterministic evaluation consistently reaches the
episode time limit with decreasing inclination and pitch-rate errors.

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
