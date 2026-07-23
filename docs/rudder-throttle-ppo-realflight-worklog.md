# Rudder + throttle PPO RealFlight validation

Date: 2026-07-24

## Trainer and episode setup

- RealFlight Airplane Hover Trainer
- Enabled controls: rudder and throttle only
- PPO outputs: `[rudder, throttle]`
- Aileron and elevator transmitted at neutral
- Artificial episode boundary: RFLink close/reconnect every 300 steps

The measured rudder polarity is unchanged from the rudder-only experiment:
positive rudder produces negative yaw rate. The integrated rudder angle is
therefore corrected by a same-sign rudder output. Throttle uses target-relative
AGL and up-positive vertical velocity.

The initial AGL target remains fixed across deliberate connection boundaries,
so a new episode cannot hide height drift. An actual trainer reposition creates
a new target. The integrated rudder angle is episode-relative and resets to zero
after each reconnect because vertical Euler angles do not provide a reliable
absolute yaw-axis reference.

## Policy and reward

The four observations are:

1. normalized integrated rudder-angle error;
2. normalized yaw rate;
3. normalized AGL error;
4. normalized up-positive vertical velocity.

The actor reuses the two minimal structured controllers:

- rudder latent =
  `k_angle * angle_error + k_rate * yaw_rate`
- throttle latent =
  `trim - k_altitude * altitude_error - k_velocity * vertical_velocity`

All four gains remain positive by construction. The reward uses only squared
normalized rudder-angle, yaw-rate, altitude, vertical-velocity, rudder-change,
and throttle-change penalties plus the survival reward. The unused planar
boundary penalty is zero.

## Actual PPO training

Command:

```bash
hoverpilot-ppo train --control-mode rudder-throttle \
  --timesteps 1024 --n-steps 256 --batch-size 64 --epochs 5 \
  --max-episode-steps 300 --eval-episodes 3 --seed 29 \
  --telemetry-log-interval-steps 25 --disable-tensorboard \
  --save-path ppo_hoverpilot_rudder_throttle.pt
```

Results:

- Training completed 1,024 steps with deliberate reconnects at every
  300-step boundary.
- Completed training episodes scored `299.475`, `299.698`, and `299.620`
  out of 300.
- Final evaluation completed 3/3 episodes at 300 steps.
- Evaluation average reward was `299.851 / 300`, or `1.000` per step.
- Rudder recovery-probe minimum margin was `1.200`.
- Throttle recovery-probe minimum margin was `1.500`.
- Both recovery probes retained zero symmetry error.
- Logged AGL error during training was generally within about `0.12 m` and
  settled close to the original target after reconnects.

The deterministic saved checkpoint then completed 5/5 episodes (1,500 steps).
Episode rewards were `299.883`, `299.869`, `299.897`, `299.687`, and `299.807`.
After each reconnect transient, logged AGL converged to approximately
`1.948–1.950 m`.

## Simplification and regression review

- Reused existing rudder and throttle feature types, observation scaling,
  structured actor parameters, action transforms, and recovery probes.
- Added only one four-value observation composition helper; no new estimator,
  MLP action head, integral altitude controller, or compatibility layer was
  introduced.
- Connection establishment uses neutral rudder and the known throttle trim
  `0.65`, avoiding an initial control discontinuity.
- Tests cover reward terms, observation order, action mapping, both restoring
  directions, initial actions, checkpoint scales, target persistence, and
  rudder-angle reset behavior.
- Final regression result: `171 passed, 28 subtests passed`.
- The unchanged checkpoint completed a post-review 3/3 episodes with rewards
  `299.887`, `299.791`, and `299.922`; AGL again converged to about `1.950 m`,
  confirming no performance degradation.
