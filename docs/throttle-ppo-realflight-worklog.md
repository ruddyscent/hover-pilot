# Throttle-only PPO RealFlight validation

Date: 2026-07-24
Branch: `codex/throttle-hover-control`

## RealFlight setup and measured response

- Trainer: Airplane Hover Trainer
- Enabled control: throttle only
- AGL target: `1.5 m`
- Episode length: 300 steps
- Episode boundary: deliberate RFLink close and reconnect

Measured throttle response established the control polarity:

- throttle `0.80` raised AGL from `1.31 m` to `7.68 m`;
- throttle `0.30` lowered AGL from `12.99 m` to `1.45 m`;
- RealFlight `m_velocityWorldW_MPS` was negative while climbing and positive
  while descending.

The low-altitude hover throttle was approximately `0.65`. The trainer prevents
a normal ground collision in this mode, so no crash/reset event is available
as an episode boundary.

## Observation, reward, and policy

The two-element observation is:

```text
[
    (altitude_agl - 1.5 m) / 1.5 m,
    upward_vertical_velocity / 5 m/s
]
```

The throttle reward contains only:

- squared normalized AGL error;
- squared normalized vertical velocity;
- throttle action change.

Position, attitude, body-rate, and planar-boundary proximity penalties are
zero. The `1.5 m` target remains fixed when RFLink reconnects; only the
irrelevant planar origin is re-anchored.

The PPO actor has a learned hover trim and two positive learned gains:

```text
latent_throttle =
    trim
    - k_altitude * normalized_altitude_error
    - k_velocity * normalized_upward_velocity
```

The tanh-squashed output is mapped to `[0, 1]`. The positive-gain
parameterization guarantees that high or rising states reduce throttle and low
or falling states increase it. There is no throttle MLP, integral controller,
or reset-state altitude offset.

## Final PPO run

```bash
uv run --no-sync hoverpilot-ppo train \
  --control-mode throttle \
  --timesteps 1024 \
  --n-steps 512 \
  --batch-size 64 \
  --epochs 5 \
  --max-episode-steps 300 \
  --eval-episodes 3 \
  --seed 11 \
  --save-path ppo_hoverpilot_throttle.pt
```

Observed results:

- first episode lifted from the trainer floor to the fixed target and scored
  `299.450`;
- later training episodes scored `299.896` and `299.898`;
- all segments reached the 300-step time limit;
- recovery-probe symmetry error: `0.000`;
- minimum restoring margin after PPO updates: `1.501`;
- final evaluation average reward: `299.915`;
- final evaluation average length: `300.0`;
- no collision or terminal failure occurred.

The generated `ppo_hoverpilot_throttle.pt` checkpoint is ignored by Git.

## Bidirectional disturbance recovery

The deterministic saved policy was tested after fixed throttle pulses:

| Disturbance | Initial error/velocity | Tail-50 error/velocity | Final error/velocity |
|---|---:|---:|---:|
| throttle `0.80` | `+0.161 m / +1.513 m/s` | `0.0238 m / 0.0001 m/s` | `0.0238 m / 0.0000 m/s` |
| throttle `0.40` | `-0.097 m / -0.045 m/s` | `0.0237 m / 0.0000 m/s` | `0.0237 m / 0.0000 m/s` |

Both 300-step recoveries completed without termination.

## Simplification and degradation review

The final implementation stores only the previous throttle action required by
the smoothness reward. A shared throttle-feature helper supplies observation,
reward, telemetry, and info paths. It does not infer control mode from
observation dimensions and adds no legacy compatibility path.

The complete regression suite passed after cleanup:

```text
151 passed, 28 subtests passed
```

The same saved checkpoint was then replayed through the cleaned path for three
300-step reconnect episodes. Rewards were `299.440`, `299.919`, and `299.914`.
The first episode again included the lift from the trainer floor; the following
steady-state scores matched the pre-cleanup evaluation, showing no performance
degradation.
