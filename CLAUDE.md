# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Main repository for "Pochi", a quadruped robot for the CoRE strider division. The active code is `pochi_rl/`, a standalone uv-managed reinforcement-learning project (MuJoCo/mjlab + RSL-RL PPO). The top-level README also documents ROS 2 launch/build commands, but those apply to a separate `~/ros2_ws` colcon workspace, not code in this repo.

## Development Rules (from top-level README)

- No direct commits to `main`; work on a branch and merge via Pull Request.
- Branch names: `feature/<name>` for features, `fix/<name>` for bug fixes.
- Package names: `pochi_<name>` (so tab completion lists them together).
- CI runs `ament_clang_format` on push and may auto-commit formatting to the branch (ROS packages; not applicable to `pochi_rl` Python code).

## Commands (run inside `pochi_rl/`)

```bash
uv sync && uv pip install -e .        # setup

uv run pytest                          # all tests (conftest sets MUJOCO_GL=egl)
uv run pytest tests/test_env_smoke.py::test_env_random_rollout_nan_free  # single test

uv run ruff check src tests scripts   # lint (line-length 88, rules E,F,I,UP,B)
uv run ruff format src tests scripts  # format
uv run mypy src                       # type check

uv run python scripts/view_mjcf.py    # open robot in MuJoCo viewer

uv sync --group cad                   # extra deps for the CAD tooling
uv run python scripts/glb_to_mjcf.py --report   # regenerate MJCF + meshes from the CAD export
                                               # (--source glb falls back to the GLB)
uv run onshape-to-robot onshape/      # export CAD from Onshape to MJCF (see onshape/README.md)

# Train (single GPU; multi-GPU: add --gpu-ids "[0,1]")
MUJOCO_GL=egl uv run train Pochi-Velocity-Flat-v0 \
  --env.scene.num-envs 4096 --agent.max-iterations 1500 \
  --agent.experiment-name pochi_velocity_flat

# Playback a checkpoint
uv run play Pochi-Velocity-Flat-v0 \
  --checkpoint-file logs/rsl_rl/pochi_velocity_flat/<ts>/model_1500.pt --num-envs 1

# Export to TorchScript/ONNX
uv run python scripts/export_policy.py --checkpoint <ckpt.pt> --output exports/
```

`train`/`play` are mjlab CLI entry points; mjlab is pinned to a specific git rev in `pyproject.toml` (`[tool.uv.sources]`). Python is pinned to 3.11.

## Code Style

Python uses **2-space indentation** throughout (`indent-width = 2` in `[tool.ruff]`, line length 88). Match this — do not introduce 4-space indentation.

## Architecture

The project is deliberately split into a backend-neutral layer and backend-specific layers so an Isaac Lab backend (Phase 2) can be added later without touching Phase 1:

- **`src/pochi_rl/task_spec.py`** — backend-neutral single source of truth for the velocity task: command ranges, reward weights, observation noise, domain-randomization event ranges, and control timing (`sim_dt=0.005`, `decimation=4` → 50 Hz policy). `POCHI_TASK_SPEC` is consumed by the env config; tune task parameters here, not in the env config.
- **`src/pochi_rl/robot/pochi_constants.py`** — single source of truth for joint/body/geom naming and action order (4 legs × 3 DoF: `hip_roll`, `hip_pitch`, `knee`) plus the CAD-derived link lengths and the default stance. Names must stay in sync with the MJCF.
- **`src/pochi_rl/mjlab/`** — mjlab backend:
  - `entity_cfg.py` wraps `assets/pochi/pochi.xml` into an mjlab `EntityCfg`.
  - `velocity_flat_env_cfg.py` builds the `ManagerBasedRlEnvCfg` (observations, rewards, events, terminations, contact sensors) from `POCHI_TASK_SPEC`; `play=True` produces the single-env playback variant.
  - `agents/rsl_rl_ppo_cfg.py` is the PPO runner config.
  - `tasks.py` registers `Pochi-Velocity-Flat-v0` in mjlab's task registry. Registration is triggered by importing `pochi_rl` (also wired via the `mjlab.tasks` entry point in `pyproject.toml`), which is why the editable install is required.
  - `runner.py` (`PochiOnPolicyRunner`) contains multi-GPU workarounds for the torchrunx/NCCL stack: skips rsl-rl's initial parameter broadcast (seeds are normalized per rank instead), reduces gradients over a gloo group, and guards atexit against torch-elastic signal exceptions. Be careful when upgrading rsl-rl/mjlab — these override internals by monkey-patching.
- **`src/pochi_rl/isaaclab/`** — Phase 2 stub only (see its `STUB.md`).
- **`src/pochi_rl/cad/`** — CAD import. `urdf.py` reads the onshape-to-robot export (`pkg_4leg_assem2/`) and is the primary input: it cuts the flat export's ~1600 spurious mate-joints down to the twelve `dof_<LEG>_<KIND>` mates and takes Onshape's own part masses. `glb.py` is a minimal glTF-binary reader used by the `--source glb` fallback, which infers the joint axes from the twelve RS02 placements instead. `convert.py` normalises every link into a canonical zero pose, computes mass properties, and emits the MJCF plus decimated visual meshes. Driven by `scripts/glb_to_mjcf.py`.
- **`assets/pochi/`** — **generated**: `pochi.xml` and `meshes/*.obj` come from `scripts/glb_to_mjcf.py`; edit the converter and regenerate, never the XML. `scene_flat.xml` (hand-written) adds the ground plane. Conventions, CAD-derived dimensions, and the list of remaining assumptions (RS02 mass, joint limits, actuator gains) are in `assets/pochi/README.md`. Visual geoms are CAD meshes; collision geoms are primitives only.
- **`tests/`** — fast checks: MJCF loads with expected joint/body counts and mass, task-spec invariants, a 4-env CPU rollout NaN check, and a CAD round-trip (driving the generated model to the CAD joint angles must rebuild the original part placements). Tests skip gracefully if `mujoco`/`mjlab`/the CAD export are missing.

Training outputs go to `logs/rsl_rl/<experiment>/<timestamp>/` (checkpoints, tensorboard, params dump) and are gitignored.
