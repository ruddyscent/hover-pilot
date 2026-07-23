# Aileron-only PPO RealFlight validation

Date: 2026-07-24
Branch: `codex/aileron-hover-roll-control`

## RealFlight setup

- Trainer: Airplane Hover Trainer
- Enabled control: aileron only
- RFLink endpoint: project `.env` configuration
- Episode boundary: deliberate RFLink close and reconnect at the time limit
- Episode length used for the final run: 300 steps

## Control polarity

A conservative `±0.10` aileron pulse established the measured sign:

- positive aileron moved roll rate in the positive direction;
- negative aileron moved roll rate in the negative direction.

The controller correction therefore uses the opposite sign of normalized roll
error and roll rate.

## Reward and policy

The aileron reward contains only terms that the active axis can influence:

- wrapped target-relative roll error;
- roll rate;
- aileron action change.

Position, altitude, and velocity penalties are zero in the aileron reward
profile. The policy has three learned scalar parameters: a trim and two
positive gains. Its latent mean is:

```text
trim - k_roll * normalized_roll_error - k_rate * normalized_roll_rate
```

The trim is necessary because the trainer produced an approximately constant
negative roll disturbance. A gain-only controller reduced roll rate but
settled with a large roll-angle offset. Adding one trim parameter removed that
steady-state error without introducing an integrator or another network.

## Episode lifecycle finding

RealFlight reported `controller_active=0` and `engine_running=0` even for the
valid, stationary aileron hover. The crash-oriented lifecycle heuristic
therefore misclassified a stabilized aircraft as a pre-reset wait state.
Aileron mode now treats that stationary state as valid; closing and reopening
RFLink is the episode boundary.

The final training run completed repeated close/connect cycles at steps 300,
600, and 900. Final evaluation completed three additional 300-step episodes
using the same reconnect boundary.

## Final run

Command:

```bash
uv run --no-sync hoverpilot-ppo train \
  --control-mode aileron \
  --timesteps 1024 \
  --n-steps 256 \
  --batch-size 64 \
  --epochs 5 \
  --max-episode-steps 300 \
  --eval-episodes 3 \
  --seed 42 \
  --save-path ppo_hoverpilot_aileron.pt
```

Observed results:

- first episode recovered from an initial roll rate near `-60 deg/s`;
- at step 300: roll error about `-4.3 deg`, roll rate about `+1.7 deg/s`;
- later training episodes stayed mostly within `1 deg` roll error and
  `1 deg/s` roll rate;
- later 300-step episode rewards were `299.867` and `299.727`;
- final evaluation: average reward `299.841`, average length `300.0`, reward
  per step `0.999`;
- no collision/reset termination occurred;
- the analytic recovery probe retained zero symmetry error and positive
  correction margins after every PPO update.

Checkpoint: `ppo_hoverpilot_aileron.pt` (ignored by Git as a generated model).

## Simplification and regression review

The first gain-only version was rejected because it left roughly `38 deg` of
steady-state roll error. The final controller adds only the single trim scalar
needed by the measured persistent disturbance. No aileron MLP, integral-state
machinery, position recovery logic, or simulator-specific reset emulation was
added.

The final full test run passed:

```text
137 passed, 26 subtests passed
```

This includes the pre-existing standard/elevator coverage plus aileron reward,
observation, stationary reconnect, sign-preserving policy, and checkpoint
round-trip tests.
