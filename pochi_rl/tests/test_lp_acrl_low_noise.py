"""Low-noise task remains a real conditional Gaussian and rejects mixed contracts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from test_lp_acrl_contact_runner import fixture as contact_fixture
from torch.distributions import Normal

from pochi_rl.mjlab.lp_acrl_low_noise import LPACRLLowNoiseOnPolicyRunner
from pochi_rl.mjlab.lp_acrl_resume import _equal_tree


def fixture(tmp_path, monkeypatch):
  base, saved, path, _ = contact_fixture(tmp_path, monkeypatch)
  runner = LPACRLLowNoiseOnPolicyRunner.__new__(LPACRLLowNoiseOnPolicyRunner)
  runner.__dict__.update(base.__dict__)
  runner.env.unwrapped.cfg = SimpleNamespace(observations={
    name: SimpleNamespace(enable_corruption=False) for name in ("actor", "critic")
  })
  distribution = runner.alg._raw_actor.distribution
  distribution.std_range = [.03, .03]
  distribution.std_param.requires_grad_(False)
  distribution.std_param.fill_(.03)
  runner.alg.symmetry.use_data_augmentation = False
  saved["actor_state_dict"]["distribution.std_param"].fill_(.03)
  saved["infos"]["lp_acrl_training_exploration"] = runner.exploration_identity()
  saved["infos"]["lp_acrl_training_symmetry"] = runner.symmetry_identity()
  saved["infos"]["lp_acrl_observation_corruption"] = (
    runner.observation_corruption_identity()
  )
  torch.save(saved, path)
  return runner, saved, path


def test_fixed_conditional_gaussian_likelihood_and_mean_are_consistent(
  tmp_path, monkeypatch,
):
  runner, _, _ = fixture(tmp_path, monkeypatch)
  actor = runner.alg._raw_actor
  obs = TensorDict({"actor": torch.zeros(2, 48)}, batch_size=[2])
  obs["actor"][1, 9] = .25
  mean = actor(obs).detach().clone()
  actions = actor(obs, stochastic_output=True)
  expected_std = torch.tensor([.01, .03])[:, None].expand(2, 12)
  torch.testing.assert_close(actor.output_std, expected_std)
  expected = Normal(mean, expected_std)
  torch.testing.assert_close(
    actor.get_output_log_prob(actions), expected.log_prob(actions).sum(-1)
  )
  actor(obs, stochastic_output=True)
  torch.testing.assert_close(
    actor.get_output_log_prob(actions), expected.log_prob(actions).sum(-1)
  )
  torch.testing.assert_close(actor(obs), mean)
  assert not actor.distribution.std_param.requires_grad


@pytest.mark.parametrize("actor_only", [False, True])
@pytest.mark.parametrize(
  "change", ["missing", "corruption", "exploration", "symmetry", "std", "nan"]
)
def test_wrong_saved_contract_rejects_before_weights_or_lp_change(
  tmp_path, monkeypatch, actor_only, change,
):
  runner, saved, path = fixture(tmp_path, monkeypatch)
  if change == "missing":
    saved["infos"].pop("lp_acrl_observation_corruption")
  elif change == "corruption":
    saved["infos"]["lp_acrl_observation_corruption"]["groups"]["actor"] = True
  elif change == "exploration":
    saved["infos"]["lp_acrl_training_exploration"]["moving_std_learned"] = True
  elif change == "symmetry":
    saved["infos"]["lp_acrl_training_symmetry"]["use_data_augmentation"] = True
  else:
    saved["actor_state_dict"]["distribution.std_param"].fill_(
      float("nan") if change == "nan" else .12
    )
  torch.save(saved, path)
  before = deepcopy((
    runner.alg._raw_actor.state_dict(), runner.command_term.curriculum_state_dict()
  ))
  with pytest.raises(ValueError):
    runner.load(str(path), {"actor": True} if actor_only else None)
  assert _equal_tree(before, (
    runner.alg._raw_actor.state_dict(), runner.command_term.curriculum_state_dict()
  ))


@pytest.mark.parametrize(
  "change", ["corruption", "range", "learned", "symmetry", "std"]
)
def test_runtime_contract_changes_are_rejected(tmp_path, monkeypatch, change):
  runner, _, _ = fixture(tmp_path, monkeypatch)
  distribution = runner.alg._raw_actor.distribution
  if change == "corruption":
    runner.env.unwrapped.cfg.observations["actor"].enable_corruption = True
  elif change == "range":
    distribution.std_range = [.01, .1]
  elif change == "learned":
    distribution.std_param.requires_grad_(True)
  elif change == "symmetry":
    runner.alg.symmetry.use_data_augmentation = True
  else:
    distribution.std_param.fill_(.1)
  with pytest.raises(ValueError):
    runner.validate_low_noise_configuration()


def test_actual_save_and_restore_preserve_all_contracts(tmp_path, monkeypatch):
  runner, saved, source = fixture(tmp_path, monkeypatch)
  runner.load(str(source))
  assert _equal_tree(saved["actor_state_dict"], runner.alg._raw_actor.state_dict())
  runner.alg.save = lambda: PPO.save(runner.alg)
  runner.cfg = {"upload_model": False}
  runner.env.unwrapped.common_step_counter = 24
  path = tmp_path / "low_noise.pt"
  runner.save(str(path), {"marker": "kept"})
  result = torch.load(path, map_location="cpu", weights_only=False)
  assert result["infos"]["marker"] == "kept"
  for key in (
    "lp_acrl_contact_backend", "lp_acrl_training_exploration",
    "lp_acrl_training_symmetry", "lp_acrl_observation_corruption",
  ):
    assert result["infos"][key] == saved["infos"][key]
  runner.load(str(path), {"actor": True})


def test_corruption_identity_returns_independent_data(tmp_path, monkeypatch):
  runner, _, _ = fixture(tmp_path, monkeypatch)
  identity = runner.observation_corruption_identity()
  identity["groups"]["actor"] = True
  assert runner.observation_corruption_identity()["groups"]["actor"] is False
