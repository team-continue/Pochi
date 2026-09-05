# Pochi RL

Standalone uv-managed reinforcement-learning project for the Pochi quadruped.
Phase 1 targets mjlab with a starter 4-leg x 3-DoF MJCF and a flat-ground
velocity-tracking PPO task. Phase 2 can add Isaac Lab configs while reusing the
same `task_spec.py` and robot constants.

## Setup

```bash
cd /home/k_suzuki/fun_ws/Pochi/pochi_rl
uv sync
uv pip install -e .
```

## Inspect the Robot

```bash
uv run python scripts/view_mjcf.py
```

## Train

Single GPU:

```bash
MUJOCO_GL=egl uv run train Pochi-Velocity-Flat-v0 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 1500 \
  --agent.experiment-name pochi_velocity_flat
```

Multi GPU:

```bash
MUJOCO_GL=egl uv run train Pochi-Velocity-Flat-v0 \
  --env.scene.num-envs 4096 \
  --gpu-ids "[0,1]"
```

Playback:

```bash
uv run play Pochi-Velocity-Flat-v0 \
  --checkpoint-file logs/rsl_rl/pochi_velocity_flat/<ts>/model_1500.pt \
  --num-envs 1
```

Export:

```bash
uv run python scripts/export_policy.py \
  --checkpoint logs/rsl_rl/pochi_velocity_flat/<ts>/model_1500.pt \
  --output exports/
```

## Deliberate, one-foot-at-a-time walking

`Pochi-Deliberate-Walk-v0` trains a separate low-speed policy. Its commands
cover 0.06–0.14 m/s forward, up to 0.025 m/s sideways and 0.15 rad/s turning.
The target gait is RL → FL → RR → FR: a 2.8 s cycle, 0.56 s per swing and
0.14 s with all four feet down between swings. The swing reference lifts the
foot 4.5 cm with zero vertical velocity at either end. Contact timing, foot
height, support and slipping are explicit objectives; these are learned
preferences, not hard guarantees on the resulting motion.

```bash
MUJOCO_GL=egl uv run train Pochi-Deliberate-Walk-v0 \
  --gpu-ids "[1]" --env.scene.num-envs 4096 --agent.max-iterations 1500

MUJOCO_GL=egl uv run play Pochi-Deliberate-Walk-v0 \
  --checkpoint-file logs/rsl_rl/pochi_deliberate_walk/<run>/model_1499.pt \
  --num-envs 1 --viewer viser

MUJOCO_GL=egl uv run python scripts/eval_deliberate_walk.py \
  --checkpoint logs/rsl_rl/pochi_deliberate_walk/<run>/model_1499.pt \
  --output /tmp/deliberate-walk-eval.json
```

Playback starts at 0.10 m/s straight ahead. The viewer has **Stop** and
**Walk slowly** buttons and centimetre-per-second velocity controls. It reports
its actual HTTP port on startup (8080, or the next free port).

The policy observes a sine/cosine gait clock in addition to the velocity
task's 48 observations, so it needs its own 50-input checkpoint. The clock is
per-episode time in seconds; it must be supplied consistently at deployment.
The phase, lift and contact schedule are disabled at near-zero commands.

The evaluator uses the same seed and commands for both tasks. Pass
`--task Pochi-Velocity-Flat-v0` with an older checkpoint for comparison. It
reports achieved velocity, joint speeds, support fraction, contact slip,
footfalls after at least 0.10 s airborne, and falls. A motionless or sliding
policy should not be accepted merely because its joint speeds are low.

Validated local checkpoint (2026-09-05):
`logs/rsl_rl/pochi_deliberate_walk/2026-09-05_12-03-42_sequential/model_950.pt`.
It passed a 32-environment continuous sequence of 0.10 → 0 → 0.10 → 0.14 →
0.06 m/s, 30.8 s per command including settling, with no falls. At the default
0.10 m/s command it achieved 0.098 m/s, 1.44 total footfalls/s and 97% of time
with at least three feet in contact. Contact-site slip remained about 0.022 m/s;
contact timing and foot clearance are learned approximately. See
`logs/deliberate_walk/final_continuous.json` and
`logs/deliberate_walk/deliberate_walk.mp4` for the local results. Training was
stopped after this checkpoint passed evaluation; later intermediate checkpoints
are retained, but were not selected on these tests.

## Notes

- The MJCF uses primitive collision/visual geometry only. Keep collision
  primitives when CAD meshes arrive unless a task needs detailed contacts.
- Joint names and action order are defined in
  `src/pochi_rl/robot/pochi_constants.py`.
- The default control path is MuJoCo position actuators with `kp=40`,
  `forcerange="-30 30"`, and policy deltas scaled by `action_scale=0.25`.
- Masses, inertias, dimensions, joint limits, and the default crouch are starter
  values and should be replaced with CAD/hardware values before sim-to-real.
