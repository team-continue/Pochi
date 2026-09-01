# Onshape CAD Import

Exports the Pochi CAD assembly from Onshape to MJCF using
[onshape-to-robot](https://onshape-to-robot.readthedocs.io/).

## Setup (once)

1. Create API keys at <https://cad.onshape.com/user/developer/apiKeys>.
2. `cp .env.example .env` and fill in the keys (`.env` is gitignored).
3. Paste the Onshape **assembly** URL into `config.json` (`url`).

## Export

```bash
cd pochi_rl
uv run onshape-to-robot onshape/
```

Outputs `onshape/pochi_cad.xml` plus meshes in `onshape/assets/`. Preview with:

```bash
uv run onshape-to-robot-mujoco onshape/
```

## Required conventions in the Onshape assembly

onshape-to-robot derives the robot structure from mate-connector names
(see the [design page](https://onshape-to-robot.readthedocs.io/en/latest/design.html)).
Names must match `src/pochi_rl/robot/pochi_constants.py` so the exported model
is a drop-in for the training pipeline:

- The **first instance** in the top-level assembly is the base link. Attach a
  mate connector named `link_base_link` to name it `base_link`.
- **Joints** (12): revolute mates named `dof_<LEG>_<KIND>` for each leg
  `FL, FR, RL, RR` and kind `hip_roll, hip_pitch, knee`
  (e.g. `dof_FL_hip_roll`). Joints rotate around the mate's z-axis; append
  `_inv` (e.g. `dof_FL_knee_inv`) where needed to match the sign conventions
  in `assets/pochi/README.md`: hip pitch positive swings the leg backward,
  knee positive swings the shank backward, hip roll positive swings
  both legs to the robot's left.
- **Sites**: mate connectors `frame_FL_foot_site`, `frame_FR_foot_site`,
  `frame_RL_foot_site`, `frame_RR_foot_site` at the foot centers, and
  `frame_imu` on the torso. These become MuJoCo sites.
- Frames: torso x-forward, y-left, z-up.

## Integration into training assets

Per project policy (`assets/pochi/README.md`), CAD meshes replace **visual**
geoms only; the primitive **collision** geoms in `assets/pochi/pochi.xml`
(`<LEG>_foot_collision`, `<LEG>_thigh_collision`, `<LEG>_shank_collision`,
`base_collision`) stay, because the contact sensors and rewards in
`velocity_flat_env_cfg.py` reference them by name. After a satisfactory
export:

1. Compare masses/inertias/joint placements against `pochi.xml` and update the
   hand-written model with the CAD values.
2. Copy meshes into `assets/pochi/meshes/` and reference them as visual geoms.
3. `uv run pytest` — `test_mjcf_loads.py` pins joint/actuator/body counts.
