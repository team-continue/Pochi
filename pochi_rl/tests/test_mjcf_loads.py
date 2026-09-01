from pathlib import Path

import pytest

from pochi_rl.robot import FOOT_GEOMS, JOINT_NAMES


def _model():
  mujoco = pytest.importorskip("mujoco")
  return mujoco, mujoco.MjModel.from_xml_path(
    str(Path("assets") / "pochi" / "scene_flat.xml")
  )


def test_mjcf_loads() -> None:
  _mujoco, model = _model()
  assert model.njnt == 13
  assert model.nu == 12
  # world + base_link + 4 legs * (hip, thigh, shank, foot)
  assert model.nbody == 18


def test_mjcf_exposes_expected_names() -> None:
  mujoco, model = _model()
  for name in JOINT_NAMES + FOOT_GEOMS:
    obj = mujoco.mjtObj.mjOBJ_JOINT if name in JOINT_NAMES else mujoco.mjtObj.mjOBJ_GEOM
    assert mujoco.mj_name2id(model, obj, name) >= 0, name


def test_mass_matches_cad_estimate() -> None:
  _mujoco, model = _model()
  # 10.92 kg: Onshape's own part masses (via the onshape-to-robot URDF), the
  # datasheet 0.405 kg RS02, and the 3 kg torso ballast.  Guards against a
  # silent regression in the generated inertials -- and against the URDF mass
  # table quietly falling back to the density guess, 0.8 kg heavier.
  from pochi_rl.robot import PAYLOAD_MASS_KG

  assert model.body_subtreemass[1] == pytest.approx(7.923 + PAYLOAD_MASS_KG, abs=0.05)


def test_xml_actuators_carry_the_rs02_peak_torque() -> None:
  """The standalone model is driven by its own <position> actuators.

  mjlab replaces these (see ``pochi_rl.mjlab.entity_cfg.pochi_spec``), but the
  MJCF has to stand on its own for the viewer and the ROS side, so the peak
  torque still has to be right here.
  """
  _mujoco, model = _model()
  from pochi_rl.robot import RS02_PEAK_TORQUE_NM

  assert model.nu == 12
  assert model.actuator_forcelimited.all()
  for i in range(model.nu):
    lo, hi = model.actuator_forcerange[i]
    assert (lo, hi) == pytest.approx((-RS02_PEAK_TORQUE_NM, RS02_PEAK_TORQUE_NM))


def test_default_stance_puts_the_feet_on_the_ground() -> None:
  """``NOMINAL_BASE_HEIGHT`` must match the generated collision geometry.

  The foot pad is a mesh taken from the CAD, so its lowest point moves whenever
  the pad or the stance changes; a stale constant would spawn the robot either
  floating or already penetrating the floor.
  """
  mujoco, model = _model()
  from pochi_rl.robot import DEFAULT_JOINT_POS, NOMINAL_BASE_HEIGHT

  data = mujoco.MjData(model)
  for name, value in DEFAULT_JOINT_POS.items():
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint]] = value
  data.qpos[2] = NOMINAL_BASE_HEIGHT
  mujoco.mj_forward(model, data)

  lowest = min(
    (
      (
        model.mesh_vert[
          model.mesh_vertadr[model.geom_dataid[g]] : model.mesh_vertadr[
            model.geom_dataid[g]
          ]
          + model.mesh_vertnum[model.geom_dataid[g]]
        ]
        @ data.geom_xmat[g].reshape(3, 3).T
        + data.geom_xpos[g]
      )[:, 2].min()
      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
      else data.geom_xpos[g][2] - model.geom_rbound[g]
    )
    for g in range(model.ngeom)
    if model.geom_group[g] == 3
  )
  assert lowest == pytest.approx(0.0, abs=1e-3)
