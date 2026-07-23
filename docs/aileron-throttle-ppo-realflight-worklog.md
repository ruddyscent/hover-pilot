# Aileron + throttle PPO RealFlight validation

Date: 2026-07-24

## Trainer and episode setup

- RealFlight Airplane Hover Trainer
- Enabled controls: aileron and throttle only
- PPO outputs: `[aileron, throttle]`
- Elevator and rudder transmitted at neutral
- Artificial episode boundary: RFLink close/reconnect every 300 steps

The previously measured aileron polarity remains applicable: positive aileron
produces positive roll rate. The restoring correction therefore opposes the
weighted roll error and roll rate. Throttle uses target-relative AGL and an
up-positive vertical velocity.

The initial trainer roll and AGL become the targets. They remain fixed across
deliberate connection boundaries so that a new episode cannot hide drift. An
actual trainer reposition creates a new local target.

## Policy and reward

The four observations are:

1. normalized wrapped roll error;
2. normalized roll rate;
3. normalized AGL error;
4. normalized up-positive vertical velocity.

The actor reuses the two minimal structured controllers:

- `aileron = trim - k_roll * roll_error - k_rate * roll_rate`
- throttle latent =
  `trim - k_altitude * altitude_error - k_velocity * vertical_velocity`

All four gains remain positive by construction. The reward uses only squared
normalized roll, roll-rate, altitude, vertical-velocity, aileron-change, and
throttle-change penalties plus the survival reward. The unused planar boundary
penalty is zero.

## Actual PPO training

Command:

```bash
hoverpilot-ppo train --control-mode aileron-throttle \
  --timesteps 1024 --n-steps 256 --batch-size 64 --epochs 5 \
  --max-episode-steps 300 --eval-episodes 3 --seed 23 \
  --disable-tensorboard \
  --save-path ppo_hoverpilot_aileron_throttle.pt
```

Results:

- The initial approximately `-16.7°` roll error recovered toward zero.
- Rollout mean step reward improved from `0.891` to `0.998`.
- The final training episode reward was `298.830 / 300`.
- Final evaluation completed 3/3 episodes.
- Evaluation reward per step was `0.998`.
- Aileron recovery-probe minimum margin was `0.800`.
- Throttle recovery-probe minimum margin was `1.500`.
- Both recovery probes retained zero symmetry error.
- AGL stayed about `0.03 m` from the original target across reconnects,
  demonstrating that reconnect did not re-anchor altitude.

The deterministic saved checkpoint then completed 5/5 episodes (1,500 steps).
Episode rewards were `281.493`, `294.053`, `296.782`, `297.212`, and `297.310`;
the lower first result includes initial roll settling. Logged AGL remained
approximately `1.34–1.37 m`.

After the simplification pass, the same checkpoint completed another 3/3
episodes with rewards `290.654`, `293.981`, and `294.631`; AGL held at about
`1.310 m`. This confirmed that the cleanup did not degrade episode completion
or altitude stability.

## Simplification and regression review

- Reused existing aileron and throttle feature types, observation scaling,
  structured actor parameters, action transforms, and recovery probes.
- Added only one four-value observation composition helper; no additional
  state estimator, MLP actor, integral controller, or compatibility layer was
  introduced.
- The actor sends the known aileron trim `0.78` and throttle trim `0.65` while
  establishing a connection boundary, avoiding a control discontinuity.
- Tests cover reward terms, observation order, action mapping, both restoring
  directions, initial actions, checkpoint scales, and target persistence.
- Final regression result: `164 passed, 28 subtests passed`.
