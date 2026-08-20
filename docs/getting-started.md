# Getting Started with HoverPilot

This guide takes you from installation to a first elevator PPO run. Commands
that send flight controls are labeled explicitly.

## Before You Begin

You need:

- Python 3.10 or newer
- RealFlight running with RealFlight Link enabled on TCP port `18083`
- an Airplane Hover Trainer scenario loaded in RealFlight
- the host or IP address of the RealFlight computer

When RealFlight is on the same computer, the default host is `127.0.0.1`. When
it is elsewhere, set the reachable address before continuing:

```bash
export RFLINK_HOST=<realflight-host-ip>
export RFLINK_PORT=18083
```

## 1. Install

Install the command-line tools and reinforcement-learning dependencies:

```bash
python -m pip install "hover-pilot[rl]"
```

For a source checkout instead:

```bash
uv sync --extra rl
cp .env.example .env
```

Prefix the remaining commands with `uv run` when using a source checkout.

## 2. Check the Connection

This check does not inject a controller or send flight controls:

```bash
hoverpilot-doctor
```

Expected result:

```text
[DOCTOR] OK: TCP endpoint is reachable.
[DOCTOR] No controller was injected and no flight controls were sent.
```

If it fails, check that RealFlight and RealFlight Link are running, port 18083
is reachable, and `RFLINK_HOST` points to the RealFlight computer.

## 3. Validate Safely

The validator injects the HoverPilot controller but sends only neutral controls
with zero throttle by default:

```bash
hoverpilot-validate --episodes 1 --max-episode-steps 50
```

It prints the action and observation spaces, episode readiness, rewards, and
termination reason. A failed check returns a non-zero process status.

`--control-test` sends random flight controls and can move or crash the simulated
aircraft. Use it only when intentionally testing action scaling.

## 4. Run the Demo

The demo sends neutral aileron, elevator, and rudder with throttle `0.55`.
It stops after 100 control steps:

```bash
hoverpilot-demo
```

Use `hoverpilot-demo --help` to change the step limit, throttle, host, or port.
Continuous execution requires the explicit `--forever` option.

## 5. Create a Starter Experiment

Generate a local, editable elevator configuration:

```bash
hoverpilot-ppo init-config
```

This creates `hoverpilot-elevator.toml` without overwriting an existing file.
Open it to confirm the RFLink host, output paths, and training duration.

## 6. Train and Inspect Results

Start training:

```bash
hoverpilot-ppo train --config hoverpilot-elevator.toml
```

In another terminal, monitor metrics:

```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006`. Training stores the latest checkpoint at
`checkpoints/elevator.pt` and the best evaluation checkpoint at
`checkpoints/elevator.best.pt`.

Evaluate and generate a report:

```bash
hoverpilot-ppo evaluate --checkpoint checkpoints/elevator.best.pt
hoverpilot-ppo report runs/elevator
```

## Where to Go Next

- [User guide](user-guide.md): control modes, lifecycle, configuration, evaluation, and troubleshooting
- [RealFlight Link interface](realflight-link-interface.md): protocol and client API reference
- [README](../README.md): project overview and installation alternatives
