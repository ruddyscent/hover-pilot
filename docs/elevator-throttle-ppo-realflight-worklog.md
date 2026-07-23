# Elevator + throttle PPO RealFlight validation

Date: 2026-07-24

## Trainer and control setup

- RealFlight Airplane Hover Trainer
- Enabled controls: elevator and throttle only
- RFLink: `10.211.55.3:18083`
- PPO policy outputs: `[elevator, throttle]`
- Transmitted inactive channels: aileron `0`, rudder `0`
- Episode boundary: RFLink close/reconnect every 300 steps

The measured elevator response matched the existing elevator convention:
positive elevator reduced pitch rate. The combined observation therefore uses
the existing signed elevator recovery frame and converts RealFlight's
down-positive world-W velocity to an up-positive throttle feature.

## Implementation

- Added the `elevator-throttle` environment, reward, CLI, checkpoint, training,
  evaluation, and playback mode.
- Anchored the local x/y, heading, and AGL target at the initial trainer state.
- Preserved that target across deliberate connection episode boundaries.
- Allowed an actual trainer reposition to establish a new target.
- Used a structured PPO actor with positive learned gains:
  - elevator: inclination-tracking error and pitch rate
  - throttle: AGL error, up-positive vertical velocity, and learned trim
- Added recovery probes for both output channels.

## Training result

Command:

```bash
hoverpilot-ppo train --control-mode elevator-throttle \
  --timesteps 1024 --n-steps 256 --batch-size 64 --epochs 5 \
  --max-episode-steps 300 --eval-episodes 3 --seed 17 \
  --disable-tensorboard \
  --save-path ppo_hoverpilot_elevator_throttle.pt
```

Observed results:

- Rollout mean step reward improved from `0.987` to `0.997`.
- Final evaluation: 3/3 episodes reached 300 steps.
- Evaluation reward per step: `0.996`.
- Elevator recovery probe minimum margin: `0.285`.
- Throttle recovery probe minimum margin: `1.500`.
- Both probes retained zero symmetry error.
- AGL errors remained small across reconnects without re-anchoring the target.

The first separate playback inherited a transient aircraft state left by the
previous process and exited the trainer cylinder after 198 steps. Following the
trainer's actual reposition, the next two episodes completed. A subsequent
stable-state playback completed 5/5 episodes (1,500 steps) with episode rewards
from `297.588` to `298.991`. Logged AGL stayed between `1.949 m` and `1.953 m`
at episode checkpoints, and inclination stayed between `89.07°` and `89.97°`.
After correcting the reconnect action to use the same `0.65` hover throttle as
the actor trim, the final post-review playback completed another 3/3 episodes
with rewards `297.904`, `298.431`, and `298.967`.

## Regression and simplification review

- Added tests for combined observation polarity, reward terms, channel mapping,
  initial hover throttle, restoring policy directions, and target persistence.
- Kept one shared six-feature elevator representation instead of adding a
  parallel combined feature type.
- Reused the existing elevator and throttle structured actor parameters and
  probes.
- Removed a redundant combined-mode action branch found during review.
- Final regression result: `157 passed, 28 subtests passed`.
