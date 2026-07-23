# Rudder-only PPO RealFlight validation

Date: 2026-07-24
Branch: `codex/rudder-hover-control`

## Setup and episode lifecycle

- Trainer: Airplane Hover Trainer
- Enabled control: rudder only
- Policy action: one rudder scalar; aileron and elevator remain neutral
- Fixed transmitted throttle: `0.55`
- Episode length: 300 steps
- Episode boundary: deliberate RFLink close and reconnect

Rudder-only hover does not reliably produce a collision/reset event. HoverPilot
therefore closes RFLink at each time limit and reconnects for the next episode.
Stationary states remain valid even when RealFlight reports controller-active
and engine-running as zero.

## State, reward, and actor

An initial implementation used signed vertical inclination derived from Euler
inclination and azimuth. A rare reconnect state made that sign disagree with
yaw-rate polarity and caused the policy to amplify the error. The final
implementation instead integrates `m_yawRate_DEGpSEC` over RealFlight physics
time. Each reconnect establishes zero episode-relative rudder angle.

The two-element observation is:

```text
[rudder_angle_error / 15 deg, yaw_rate / 30 deg/s]
```

The reward penalizes only squared normalized rudder angle, yaw rate, and rudder
action change. Position, altitude, and unrelated-axis penalties are excluded.
The PPO actor has two positive learned gains:

```text
latent_rudder =
    k_angle * normalized_rudder_angle
    + k_rate * normalized_yaw_rate
```

RealFlight measurement showed that positive rudder produces negative yaw rate,
so the positive-gain structure enforces restoring action for both error signs.
There is no rudder MLP, trim, integral controller beyond the observed physical
angle, or simulator-specific reset emulation.

## Final training run

```bash
uv run --no-sync hoverpilot-ppo train \
  --control-mode rudder \
  --timesteps 1024 \
  --n-steps 512 \
  --batch-size 64 \
  --epochs 5 \
  --max-episode-steps 300 \
  --eval-episodes 3 \
  --seed 7 \
  --save-path ppo_hoverpilot_rudder.pt
```

Results:

- training episode rewards: `299.899`, `299.894`, and `299.892`;
- all training segments reached the 300-step time limit;
- recovery probe symmetry error: `0.000`;
- minimum restoring margin after PPO updates: `1.199`;
- final evaluation: average reward `299.741`, average length `300.0`, reward
  per step `0.999`;
- no collision or terminal failure occurred.

The generated checkpoint is `ppo_hoverpilot_rudder.pt` and is ignored by Git.

## Bidirectional disturbance validation

The saved PPO checkpoint was tested by applying 40 steps of fixed rudder before
returning control to its deterministic policy:

| Pulse | Disturbed angle/rate | Tail-50 mean angle/rate | Final angle/rate |
|---|---:|---:|---:|
| `+0.25` | `-4.947° / -9.896°/s` | `0.316° / 0.071°/s` | `0.289° / 0.060°/s` |
| `-0.25` | `+4.419° / +9.183°/s` | `0.043° / 0.063°/s` | `0.017° / 0.054°/s` |

Both 300-step recoveries completed without termination. Final physical
inclination was `89.68°` and `90.00°`, respectively.

## Simplification and regression review

The unreliable Euler-sign implementation was removed rather than supplemented
with heading, roll, or polarity heuristics. The remaining implementation uses
one integrated scalar, one timestamp, and the measured yaw rate. Existing
single-axis checkpoint and reconnect helpers are reused.

No legacy observation-dimension inference remains in `ActorCritic`.
`control_mode` is required explicitly, preventing the shared two-element
aileron/rudder shape from selecting the wrong actor.

The full regression suite after the redesign passed:

```text
144 passed, 28 subtests passed
```

The saved checkpoint was then replayed for three 300-step RealFlight episodes
through the simplified path. Rewards were `299.978`, `299.975`, and `299.958`;
all three reached the time limit without termination. This is slightly above
the pre-cleanup evaluation and shows no performance degradation.
