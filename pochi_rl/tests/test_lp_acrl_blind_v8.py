"""V8 changes exactly one reward and locally registers the existing runner."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v5_env_cfg import (
  pochi_lp_acrl_blind_v5_env_cfg,
  pochi_lp_acrl_blind_v5_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v8 import BLIND_V8_TASK_ID, register_blind_v8
from pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg import (
  common_hip_reward_cfg,
  pochi_lp_acrl_blind_v8_env_cfg,
  pochi_lp_acrl_blind_v8_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_common_hip_rewards import (
  HIP_PITCH_NAMES,
  moving_common_hip_pitch_deviation,
)
from pochi_rl.mjlab.lp_acrl_low_noise import LPACRLLowNoiseOnPolicyRunner


@pytest.mark.parametrize("play", [False, True])
def test_v8_adds_one_reward_and_preserves_every_other_environment_setting(play):
  source = pochi_lp_acrl_blind_v5_env_cfg(play=play)
  target = pochi_lp_acrl_blind_v8_env_cfg(play=play)
  assert (
    target.rewards.pop("moving_common_hip_pitch_deviation") == common_hip_reward_cfg()
  )
  assert (
    target.scene.terrain.spec_fn.__qualname__
    == source.scene.terrain.spec_fn.__qualname__
  )
  target.scene.terrain.spec_fn = source.scene.terrain.spec_fn
  assert target == source


def test_entire_ppo_model_noise_symmetry_and_optimizer_configuration_is_unchanged():
  assert asdict(pochi_lp_acrl_blind_v8_ppo_runner_cfg()) == asdict(
    pochi_lp_acrl_blind_v5_ppo_runner_cfg()
  )


def test_only_named_hip_pitch_common_mode_is_regularized():
  cfg = pochi_lp_acrl_blind_v8_env_cfg()
  assert "moving_default_joint_deviation" not in cfg.rewards
  reward = cfg.rewards["moving_common_hip_pitch_deviation"]
  assert reward.func is moving_common_hip_pitch_deviation
  assert reward.weight == -4.0
  asset = reward.params["asset_cfg"]
  assert (
    asset.joint_names
    == HIP_PITCH_NAMES
    == ("FL_hip_pitch", "FR_hip_pitch", "RL_hip_pitch", "RR_hip_pitch")
  )
  assert asset.preserve_order


def test_local_registration_is_idempotent_and_does_not_install_backend(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend

  def forbidden(*args, **kwargs):
    raise AssertionError("Task registration must not install physics")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V8_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v8() == BLIND_V8_TASK_ID
  registered = registry._REGISTRY[BLIND_V8_TASK_ID]
  assert register_blind_v8() == BLIND_V8_TASK_ID
  assert registry._REGISTRY[BLIND_V8_TASK_ID] is registered
  assert set(registry._REGISTRY) == set(before) | {BLIND_V8_TASK_ID}
  assert all(registry._REGISTRY[key] is value for key, value in before.items())
  assert registry.load_runner_cls(BLIND_V8_TASK_ID) is LPACRLLowNoiseOnPolicyRunner
