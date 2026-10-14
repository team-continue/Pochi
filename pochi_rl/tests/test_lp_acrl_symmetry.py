"""Physical left/right reflection and RSL-RL augmentation contract."""

from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from pochi_rl.mjlab.lp_acrl_runner import (
  pochi_lp_acrl_ppo_runner_cfg,
  validate_blind_symmetry_cfg,
)
from pochi_rl.mjlab.lp_acrl_symmetry import (
  augment_blind_symmetry,
  mirror_blind_observations,
  mirror_joint_vectors,
)
from pochi_rl.robot.pochi_constants import DEFAULT_JOINT_POS, FOOT_SITES, JOINT_NAMES


def test_reflection_distinguishes_polar_axial_and_command_vectors():
  values = torch.arange(48, dtype=torch.float32).reshape(1, 48)
  mirrored = mirror_blind_observations(values)
  assert mirrored[0, :12].tolist() == [0, -1, 2, -3, 4, -5, 6, -7, 8, 9, -10, -11]
  assert mirrored[0, 12:18].tolist() == [-15, 16, 17, -12, 13, 14]
  torch.testing.assert_close(mirror_blind_observations(mirrored), values)


def test_original_and_mirror_batches_cover_actor_critic_and_differentiable_actions():
  values = torch.randn(4, 48)
  obs = TensorDict({"actor": values, "critic": values + 0.1}, batch_size=[4])
  actions = torch.randn(4, 12, requires_grad=True)
  augmented, augmented_actions = augment_blind_symmetry(None, obs, actions)
  assert augmented.batch_size == torch.Size([8])
  for name in ("actor", "critic"):
    torch.testing.assert_close(augmented[name][:4], obs[name])
    torch.testing.assert_close(
      augmented[name][4:], mirror_blind_observations(obs[name])
    )
  torch.testing.assert_close(augmented_actions[:4], actions)
  torch.testing.assert_close(augmented_actions[4:], mirror_joint_vectors(actions))
  augmented_actions.square().sum().backward()
  torch.testing.assert_close(actions.grad, 4 * actions.detach())
  assert augment_blind_symmetry(None, None, actions)[0] is None
  assert augment_blind_symmetry(None, obs, None)[1] is None


def test_reflection_matches_mujoco_foot_kinematics_and_default_pose():
  model = mujoco.MjModel.from_xml_path(
    str(Path(__file__).resolve().parents[1] / "assets/pochi/pochi.xml")
  )
  left = mujoco.MjData(model)
  right = mujoco.MjData(model)
  positions = torch.tensor([DEFAULT_JOINT_POS[name] for name in JOINT_NAMES])
  torch.testing.assert_close(mirror_joint_vectors(positions), positions)
  positions += torch.tensor([.1, -.1, .05, -.05, .03, -.02] * 2)
  mirrored = mirror_joint_vectors(positions)
  for index, name in enumerate(JOINT_NAMES):
    address = model.jnt_qposadr[model.joint(name).id]
    left.qpos[address] = positions[index]
    right.qpos[address] = mirrored[index]
  mujoco.mj_forward(model, left)
  mujoco.mj_forward(model, right)
  ids = [model.site(name).id for name in FOOT_SITES]
  np.testing.assert_allclose(
    right.site_xpos[ids], left.site_xpos[np.asarray(ids)[[1, 0, 3, 2]]] * [1, -1, 1],
    # CAD-derived foot offsets differ left/right by under0.1mm. Incorrect
    # hip-roll signs instead produce centimeter-scale differences here.
    atol=1e-4,
  )


def test_runner_enables_both_symmetry_modes_only_for_new_blind_task():
  cfg = pochi_lp_acrl_ppo_runner_cfg(rough=True, blind=True)
  symmetry = asdict(cfg.algorithm)["symmetry_cfg"]
  assert symmetry["use_data_augmentation"] and symmetry["use_mirror_loss"]
  assert symmetry["mirror_loss_coeff"] == 1.0
  assert cfg.experiment_name == "pochi_lp_acrl_blind"
  legacy = pochi_lp_acrl_ppo_runner_cfg(rough=True)
  assert "symmetry_cfg" not in asdict(legacy.algorithm)
  assert cfg.actor.distribution_cfg["init_std"] == .35
  assert cfg.actor.distribution_cfg["std_range"] == (.10, 1.0e6)
  assert legacy.actor.distribution_cfg["init_std"] == 1.0


def test_symmetry_layout_checks_actual_actuator_names():
  from pochi_rl.mjlab.lp_acrl_env_cfg import pochi_lp_acrl_flat_env_cfg

  cfg = pochi_lp_acrl_flat_env_cfg()
  validate_blind_symmetry_cfg(cfg)
  cfg.actions["joint_pos"].actuator_names = tuple(reversed(JOINT_NAMES))
  with pytest.raises(ValueError, match="joint order"):
    validate_blind_symmetry_cfg(cfg)


@pytest.mark.parametrize("width", [45, 323])
def test_reflection_rejects_other_observation_layouts(width):
  with pytest.raises(ValueError, match="exactly48"):
    mirror_blind_observations(torch.zeros(2, width))
