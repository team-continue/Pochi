"""V11 changes exactly the declared reward terms and initial adaptive LR."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v10_env_cfg import (
  pochi_lp_acrl_blind_v10_env_cfg,
  pochi_lp_acrl_blind_v10_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v11 import BLIND_V11_TASK_ID, register_blind_v11
from pochi_rl.mjlab.lp_acrl_blind_v11_env_cfg import (
  SOURCE_LEARNING_RATE,
  pochi_lp_acrl_blind_v11_env_cfg,
  pochi_lp_acrl_blind_v11_ppo_runner_cfg,
  stride_center_reward_cfg,
)
from pochi_rl.mjlab.lp_acrl_stride_center_rewards import (
  NOMINAL_FOOT_COM_X,
  moving_top_two_stride_center,
  nominal_stride_center_offsets,
  standing_joint_velocity_bounded_all_contact,
)
from pochi_rl.robot import FOOT_SITES


@pytest.mark.parametrize("play", [False, True])
def test_exactly_the_three_reward_edits_preserve_every_other_env_setting(play):
  source = pochi_lp_acrl_blind_v10_env_cfg(play=play)
  target = pochi_lp_acrl_blind_v11_env_cfg(play=play)
  assert "moving_common_hip_pitch_deviation" not in target.rewards
  assert (
    target.rewards.pop("moving_fore_aft_stride_center") == stride_center_reward_cfg()
  )
  target.rewards["moving_common_hip_pitch_deviation"] = source.rewards[
    "moving_common_hip_pitch_deviation"
  ]
  old, new = (
    source.rewards["standing_joint_velocity"],
    target.rewards["standing_joint_velocity"],
  )
  assert new.func is standing_joint_velocity_bounded_all_contact
  assert new.weight == old.weight == -0.25
  assert new.params == old.params == {}
  new.func = old.func
  assert target.scene.terrain.spec_fn.__qualname__ == (
    source.scene.terrain.spec_fn.__qualname__
  )
  target.scene.terrain.spec_fn = source.scene.terrain.spec_fn
  assert target == source


def test_same_ppo_noise_and_symmetry_with_source_actual_lr_only():
  source = asdict(pochi_lp_acrl_blind_v10_ppo_runner_cfg())
  target = asdict(pochi_lp_acrl_blind_v11_ppo_runner_cfg())
  assert target["algorithm"]["learning_rate"] == SOURCE_LEARNING_RATE
  assert SOURCE_LEARNING_RATE == 7.593750000000004e-05
  target["algorithm"]["learning_rate"] = source["algorithm"]["learning_rate"]
  assert target == source


def test_stride_center_uses_named_actual_nominal_whole_com_offsets():
  cfg = pochi_lp_acrl_blind_v11_env_cfg()
  term = cfg.rewards["moving_fore_aft_stride_center"]
  assert term.func is moving_top_two_stride_center
  assert term.weight == -36
  assert term.params["tau_s"] == 0.25
  assert term.params["nominal_offsets_m"] == NOMINAL_FOOT_COM_X
  assert NOMINAL_FOOT_COM_X == pytest.approx(nominal_stride_center_offsets(), abs=1e-9)
  assert term.params["asset_cfg"].site_names == FOOT_SITES
  assert term.params["asset_cfg"].preserve_order
  assert "moving_default_joint_deviation" not in cfg.rewards
  assert "moving_contact_duty_variance" not in cfg.rewards


def test_local_registration_is_idempotent_without_backend_install(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_stride_center_refinement import (
    LPACRLStrideCenterOnPolicyRunner,
  )

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install physics")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V11_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v11() == BLIND_V11_TASK_ID
  registered = entries[BLIND_V11_TASK_ID]
  assert register_blind_v11() == BLIND_V11_TASK_ID
  assert entries[BLIND_V11_TASK_ID] is registered
  assert set(entries) == set(before) | {BLIND_V11_TASK_ID}
  assert all(entries[key] is value for key, value in before.items())
  assert registry.load_runner_cls(BLIND_V11_TASK_ID) is LPACRLStrideCenterOnPolicyRunner
