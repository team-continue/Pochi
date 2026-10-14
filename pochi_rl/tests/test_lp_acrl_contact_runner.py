"""Backend identity is mandatory before any checkpoint state is restored."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from test_lp_acrl_resume import fixture as resume_fixture

from pochi_rl.mjlab import lp_acrl_contact_runner as contact
from pochi_rl.mjlab.lp_acrl_resume import _equal_tree


def fixture(tmp_path, monkeypatch):
  base, saved, path, _ = resume_fixture(tmp_path)
  identity = {"patch_id": "test-backend", "versions": {"warp": "pinned"}}
  monkeypatch.setattr(contact, "active_contact_backend_identity", lambda: identity)
  runner = contact.LPACRLContactOnPolicyRunner.__new__(
    contact.LPACRLContactOnPolicyRunner
  )
  runner.__dict__.update(base.__dict__)
  runner._contact_backend = deepcopy(identity)
  saved["infos"]["lp_acrl_contact_backend"] = deepcopy(identity)
  torch.save(saved, path)
  return runner, saved, path, identity


@pytest.mark.parametrize("actor_only", [False, True])
@pytest.mark.parametrize("metadata", [None, {"patch_id": "different"}])
def test_missing_or_different_backend_rejects_before_any_mutation(
  tmp_path, monkeypatch, actor_only, metadata
):
  runner, saved, path, _ = fixture(tmp_path, monkeypatch)
  saved["infos"]["lp_acrl_contact_backend"] = metadata
  torch.save(saved, path)
  before = deepcopy((
    runner.alg._raw_actor.state_dict(), runner.alg._raw_critic.state_dict(),
    runner.alg.optimizer.state_dict(), runner.command_term.curriculum_state_dict(),
    runner.current_learning_iteration,
  ))
  cfg = {"actor": True, "optimizer": False} if actor_only else None
  with pytest.raises(ValueError, match="contact backend differs or is missing"):
    runner.load(str(path), cfg)
  after = (
    runner.alg._raw_actor.state_dict(), runner.alg._raw_critic.state_dict(),
    runner.alg.optimizer.state_dict(), runner.command_term.curriculum_state_dict(),
    runner.current_learning_iteration,
  )
  assert _equal_tree(before, after)


def test_matching_backend_restores_actual_rsl_state(tmp_path, monkeypatch):
  runner, saved, path, identity = fixture(tmp_path, monkeypatch)
  infos = runner.load(str(path))
  assert infos["lp_acrl_contact_backend"] == identity
  assert runner.current_learning_iteration == 100
  assert _equal_tree(saved["actor_state_dict"], runner.alg._raw_actor.state_dict())
  assert _equal_tree(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict())
  assert _equal_tree(
    saved["infos"]["lp_acrl"], runner.command_term.curriculum_state_dict()
  )


def test_save_preserves_inherited_and_caller_metadata(tmp_path, monkeypatch):
  runner, _, _, identity = fixture(tmp_path, monkeypatch)
  runner.alg.save = lambda: PPO.save(runner.alg)
  runner.logger = SimpleNamespace(save_model=lambda *_: None)
  runner.cfg = {"upload_model": False}
  runner.env.unwrapped.common_step_counter = 24
  path = tmp_path / "saved.pt"
  caller = {"diagnostic": "kept"}
  runner.save(str(path), caller)
  saved = torch.load(path, map_location="cpu", weights_only=False)
  infos = saved["infos"]
  assert caller == {"diagnostic": "kept"}
  assert infos["diagnostic"] == "kept"
  assert infos["lp_acrl_contact_backend"] == identity
  assert infos["lp_acrl_training_exploration"] == runner.exploration_identity()
  assert infos["lp_acrl_training_symmetry"] == runner.symmetry_identity()
  assert _equal_tree(infos["lp_acrl"], runner.command_term.curriculum_state_dict())
  assert infos["lp_acrl_checkpoint_version"] == 1
  assert infos["env_state"] == {"common_step_counter": 24}


def test_changed_active_backend_cannot_save_or_load(tmp_path, monkeypatch):
  runner, _, path, identity = fixture(tmp_path, monkeypatch)
  identity["versions"]["warp"] = "changed"
  with pytest.raises(ValueError, match="changed after runner construction"):
    runner.load(str(path), {"actor": True, "optimizer": False})
  with pytest.raises(ValueError, match="changed after runner construction"):
    runner.save(str(tmp_path / "bad.pt"))
  assert not (tmp_path / "bad.pt").exists()


def test_returned_identity_cannot_mutate_runner_contract(tmp_path, monkeypatch):
  runner, _, _, identity = fixture(tmp_path, monkeypatch)
  copied = runner.contact_backend_identity()
  copied["versions"]["warp"] = "changed"
  assert runner.contact_backend_identity() == identity
