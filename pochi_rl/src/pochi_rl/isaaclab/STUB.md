# Isaac Lab Phase 2 Stub

Add the Isaac Lab backend here without moving Phase 1 files:

1. Add `assets/pochi/pochi.urdf` with joint names matching `JOINT_NAMES`.
2. Convert the URDF to `assets/pochi/pochi.usd` through Isaac Sim.
3. Add `pochi_articulation_cfg.py`, `velocity_flat_env_cfg.py`, and matching
   RSL-RL PPO config under this package.
4. Register `Pochi-Velocity-Flat-IsaacLab-v0`.

The shared source of truth remains `pochi_rl.task_spec` and
`pochi_rl.robot.pochi_constants`.
