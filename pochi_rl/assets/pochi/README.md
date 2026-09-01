# Pochi MJCF Conventions

`pochi.xml` and everything in `meshes/` are **generated** from the Onshape CAD
export by `scripts/glb_to_mjcf.py`. Edit the converter, not the XML:

```bash
uv sync --group cad
uv run python scripts/glb_to_mjcf.py --report
```

The primary input is **`pkg_4leg_assem2/`**, the onshape-to-robot URDF export.
It carries both halves of what the model needs:

- **Kinematics**, from the twelve `dof_<LEG>_<KIND>` mates. Their axes are
  explicit, so nothing is inferred.
- **Part masses**, as Onshape computed them from the materials assigned in CAD.

The export is *flat*: onshape-to-robot turns every untagged Onshape mate into a
joint, so the file holds ~1600 spurious prismatic/cylindrical joints for bolts
sitting in holes (1629 links, 1628 joints, 516 loop closures). Those are rigid
in reality, so `pochi_rl.cad.urdf` cuts the tree only at the twelve `dof_` mates
and welds the rest into the thirteen links. Their zero *is* the assembled pose,
so forward kinematics reproduces the CAD: all four legs come out identical to
within 0.1 mm.

`4leg_assem2.glb` remains as a fallback (`--source glb`), which recovers the
joint axes from the twelve RS02 placements instead. It agrees with the URDF's
explicit mates to 0.0002° in axis direction, which is what validated the
inference in the first place.

The RS02 is the one part with no material assigned in CAD (the URDF reports
0.18 g), so it keeps the datasheet mass. `--no-urdf-masses` falls back to
guessing a density per part name; that agrees within 2 % everywhere except
`Part 1`, which is printed plastic at ~1050 kg/m³ rather than the guessed
aluminium -- 0.8 kg of the robot. The GLB carries no kinematics, so joint axes and origins are recovered
from the twelve RobStride RS02 actuators: each motor's local `+z` is its output
axis. `Body_2` carries the four hip roll motors, each `Hiproll_1` carries a hip
pitch motor, and each `Hippitch_1` carries a knee motor.

## Frames and conventions

- Torso: x forward, y left, z up. The base origin sits at the centre of the four
  hip joints, at hip height.
- Leg labels `FL, FR, RL, RR`; joint names `<LEG>_<KIND>` with kind
  `hip_roll`, `hip_pitch`, `knee`. Action order is `JOINT_NAMES` from
  `pochi_rl.robot.pochi_constants`.
- Sign convention: hip roll positive swings both legs to the robot's left, hip
  pitch positive swings the thigh backward, knee positive swings the shank
  backward relative to the thigh.
- Zero pose: every leg straight down. The CAD ships with the legs folded flat;
  each link is rotated back about its own joint axis during conversion.
- Default stance (`DEFAULT_JOINT_POS`) mirrors front and rear, matching how the
  CAD is assembled: front hip pitch `+0.8` / knee `-1.5`, rear `-0.8` / `+1.5`.
  That balances the load across the four feet (26.8 N each) and puts the COM
  over the base centre; a uniform stance shifts it 41 mm rearward.

## Geometry (from CAD)

| quantity | value |
| --- | --- |
| hip joint offset from base centre | ±0.2545 m fore/aft, ±0.070 m lateral |
| thigh (hip pitch axis → knee axis) | 0.200 m |
| shank (knee axis → foot tip) | 0.225 m |
| foot lateral offset from leg plane | 0.098 m outboard |
| foot pad | 16 mm wide, ~19 mm radius fore/aft |
| total mass | 10.92 kg (7.92 CAD + 3.0 ballast) |

The hip roll and hip pitch axes intersect, so both hip joints share an origin
and the thigh body sits at the hip body origin. All four legs are kinematically
identical — the fore/aft difference in the CAD is only how the assembly was
posed, plus the hip housings facing inboard.

## Actuators

Every joint is a RobStride RS02. The datasheet numbers live in
`src/pochi_rl/robot/pochi_constants.py` and are shared by every backend:

| quantity | value |
| --- | --- |
| peak (stall) torque | 17 N·m |
| rated (continuous) torque | 7 N·m @ 100 rpm |
| no-load speed | 410 rpm = 42.9 rad/s at the output shaft |
| gear ratio | 7.75:1 |
| mass | 0.405 kg |

`pochi.xml` gives its `<position>` actuators a flat ±17 N·m `forcerange`, which
is all a standalone MJCF can express. Under mjlab those actuators are deleted
(`pochi_rl.mjlab.entity_cfg.pochi_spec`) and replaced by `DcMotorActuatorCfg`
`<motor>` actuators, so the torque is clamped to the real torque-speed curve —
peak torque only near zero speed, falling to zero at the no-load speed — instead
of being available at any joint velocity. This is a constraint in the actuator
path, not a reward penalty. `effort_limit` is set to the peak torque, so only
the curve binds; setting it to `RS02_RATED_TORQUE_NM` would additionally enforce
the continuous rating.

## Torso ballast

The bare CAD torso is light for the machine, so `PAYLOAD_MASS_KG = 3.0` adds a
ballast block to the base link, standing in for the battery, electronics and any
trim weight the real robot carries. It sits at the base origin (the hip centre),
so the COM stays over the support polygon. It is modelled as a steel cube -- the
density sets its size and hence its rotational inertia, which a bare point mass
would leave at zero and make the torso far too easy to spin. Set the constant to
0 to get the bare CAD robot back, or move it with `PAYLOAD_POS` once the real
payload's location is known.

## Assumptions to revisit

- **RS02 mass** is the datasheet 0.405 kg (`RS02_MASS_KG`). The OpenELAB guide
  quotes 405 g ±5 g and the RobStride repo 380 g ±3 g; twelve of them are 56 % of
  the robot mass, so replace it with a weighed value.
- Other masses come from Onshape via the URDF export. `DENSITIES` in
  `src/pochi_rl/cad/convert.py` is only the fallback for when that export is
  absent; it reads 0.8 kg heavy because it assumes `Part 1` is aluminium.
- **Joint ranges are placeholders** (`JOINT_RANGES` in
  `src/pochi_rl/cad/convert.py`): the CAD encodes no end stops. They are wide
  enough to reproduce the folded CAD pose; tighten them once the hardware
  limits are known.
- **Ballast position and extent** are a placeholder: 3 kg of steel at the hip
  centre. Replace with the real battery/electronics mass and location.
- Actuator gains (`kp = 60`, `kv = 1.5`) are hand-picked, not derived. The
  mjlab reference robots instead derive them from the reflected rotor inertia
  and a target closed-loop bandwidth, which for Pochi would give roughly
  `kp = 39.5`, `kv = 2.5`.
- **Rotor inertia** (`RS02_REFLECTED_INERTIA`, the joint `armature`) is an
  estimate; the datasheet does not publish it.
- Collision geometry is primitives (box torso and hips, capsule limbs) except
  the **foot pads**, which collide as the convex hull of the CAD part itself
  (`<LEG>_foot_pad.obj`). The pad is a printed part 16 mm wide with a ~19 mm
  radius profile fore/aft, so a sphere got the lateral contact wrong; the hull
  keeps the real footprint. `convert.foot_pad_part` finds it by geometry -- the
  shank part owning the vertex farthest from the knee axis -- not by name.
  Everything else's mesh is visual-only.
