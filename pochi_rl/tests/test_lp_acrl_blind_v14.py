"""V14 adds raw-contact duty balance without replacing any existing objective."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v3_env_cfg import pochi_lp_acrl_blind_v3_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v3_rewards import moving_contact_duty_variance
from pochi_rl.mjlab.lp_acrl_blind_v13_env_cfg import (
  pochi_lp_acrl_blind_v13_env_cfg,
  pochi_lp_acrl_blind_v13_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v14 import BLIND_V14_TASK_ID, register_blind_v14
from pochi_rl.mjlab.lp_acrl_blind_v14_env_cfg import (
  SOURCE_LEARNING_RATE,
  duty_balance_reward_cfg,
  pochi_lp_acrl_blind_v14_env_cfg,
  pochi_lp_acrl_blind_v14_ppo_runner_cfg,
)


@pytest.mark.parametrize("play", [False, True])
def test_exactly_one_added_reward_preserves_full_flat_environment(play):
  source = pochi_lp_acrl_blind_v13_env_cfg(play=play)
  target = pochi_lp_acrl_blind_v14_env_cfg(play=play)
  assert len(source.rewards) == 23
  assert len(target.rewards) == 24
  added = target.rewards.pop("moving_contact_duty_variance")
  assert added == duty_balance_reward_cfg()
  assert target.rewards == source.rewards
  assert target.scene.terrain.spec_fn.__qualname__ == (
    source.scene.terrain.spec_fn.__qualname__
  )
  target.scene.terrain.spec_fn = source.scene.terrain.spec_fn
  assert target == source


def test_duty_term_exactly_reuses_frozen_v3_kernel_weight_and_parameters():
  term = duty_balance_reward_cfg()
  old = pochi_lp_acrl_blind_v3_env_cfg().rewards["moving_contact_duty_variance"]
  assert term == old
  assert term.func is moving_contact_duty_variance
  assert term.weight == -8.0
  assert term.params == {"tau_s": 1.0}


def test_qualified_swing_and_yaw_posture_and_stop_objectives_are_all_retained():
  source = pochi_lp_acrl_blind_v13_env_cfg()
  target = pochi_lp_acrl_blind_v14_env_cfg()
  assert "moving_qualified_swing_occupancy" in target.rewards
  assert "yaw_tracking_huber" in target.rewards
  assert "moving_fore_aft_stride_center" in target.rewards
  for name in source.rewards:
    assert target.rewards[name] == source.rewards[name]
  assert target.rewards["moving_qualified_swing_occupancy"].weight == -8.0
  assert target.rewards["contact_slip"].weight == -8.0


def test_only_ppo_change_is_initial_lr_from_actual_parent_checkpoint():
  source = asdict(pochi_lp_acrl_blind_v13_ppo_runner_cfg())
  target = asdict(pochi_lp_acrl_blind_v14_ppo_runner_cfg())
  assert target["algorithm"]["learning_rate"] == SOURCE_LEARNING_RATE
  assert SOURCE_LEARNING_RATE == 7.593750000000004e-05
  target["algorithm"]["learning_rate"] = source["algorithm"]["learning_rate"]
  assert target == source


def test_factories_do_not_mutate_v13_or_share_mutable_reward_parameters():
  first = pochi_lp_acrl_blind_v14_env_cfg()
  second = pochi_lp_acrl_blind_v14_env_cfg()
  first.rewards["moving_contact_duty_variance"].params["tau_s"] = 2.0
  assert second.rewards["moving_contact_duty_variance"].params == {"tau_s": 1.0}
  assert "moving_contact_duty_variance" not in pochi_lp_acrl_blind_v13_env_cfg().rewards


def test_local_registration_is_idempotent_without_installing_physics(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_duty_balance_refinement import (
    LPACRLDutyBalanceOnPolicyRunner,
  )

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install physics")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V14_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v14() == BLIND_V14_TASK_ID
  registered = entries[BLIND_V14_TASK_ID]
  assert register_blind_v14() == BLIND_V14_TASK_ID
  assert entries[BLIND_V14_TASK_ID] is registered
  assert set(entries) == set(before) | {BLIND_V14_TASK_ID}
  assert all(entries[key] is value for key, value in before.items())
  assert registry.load_runner_cls(BLIND_V14_TASK_ID) is LPACRLDutyBalanceOnPolicyRunner
