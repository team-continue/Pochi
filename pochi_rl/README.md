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

## Notes

- The MJCF uses primitive collision/visual geometry only. Keep collision
  primitives when CAD meshes arrive unless a task needs detailed contacts.
- Joint names and action order are defined in
  `src/pochi_rl/robot/pochi_constants.py`.
- The default control path is MuJoCo position actuators with `kp=40`,
  `forcerange="-30 30"`, and policy deltas scaled by `action_scale=0.25`.
- Masses, inertias, dimensions, joint limits, and the default crouch are starter
  values and should be replaced with CAD/hardware values before sim-to-real.
