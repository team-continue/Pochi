"""Local v4 registration never installs its backend or changes old entries."""

from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v2 import QUIET_STOP_ACTOR
from pochi_rl.mjlab.lp_acrl_blind_v4 import BLIND_V4_TASK_ID, register_blind_v4
from pochi_rl.mjlab.lp_acrl_qualified_swing_rewards import (
  moving_qualified_swing_occupancy,
)


def test_registration_is_explicit_idempotent_and_uses_contact_checking_runner(
  monkeypatch,
):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_contact_runner import LPACRLContactOnPolicyRunner

  def forbidden_install(*args, **kwargs):
    raise AssertionError("Registering a task must not install a contact backend")

  monkeypatch.setattr(
    warp_contact_backend, "install_contact_backend", forbidden_install
  )
  existing = dict(registry._REGISTRY)
  existing.pop(BLIND_V4_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", existing)
  before = dict(registry._REGISTRY)
  assert register_blind_v4() == BLIND_V4_TASK_ID
  registered = registry._REGISTRY[BLIND_V4_TASK_ID]
  assert register_blind_v4() == BLIND_V4_TASK_ID
  assert registry._REGISTRY[BLIND_V4_TASK_ID] is registered
  assert set(registry.list_tasks()) == set(before) | {BLIND_V4_TASK_ID}
  assert all(registry._REGISTRY[name] is value for name, value in before.items())
  assert registry.load_runner_cls(BLIND_V4_TASK_ID) is LPACRLContactOnPolicyRunner
  assert registry.load_rl_cfg(BLIND_V4_TASK_ID).actor.class_name == QUIET_STOP_ACTOR
  for play in (False, True):
    cfg = registry.load_env_cfg(BLIND_V4_TASK_ID, play=play)
    assert "moving_contact_duty_variance" not in cfg.rewards
    assert cfg.rewards["contact_slip"].weight == -8
    assert (
      cfg.rewards["moving_qualified_swing_occupancy"].func
      is moving_qualified_swing_occupancy
    )
