"""Conditional exploration must be consistent during PPO and absent at inference."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch.distributions import Normal, kl_divergence

from pochi_rl.mjlab.lp_acrl_quiet_stop import (
  LPACRLQuietStopOnPolicyRunner,
  QuietStopMLPModel,
)
from pochi_rl.mjlab.lp_acrl_runner import LPACRLOnPolicyRunner
from pochi_rl.mjlab.lp_acrl_symmetry import augment_blind_symmetry


def models(count=8):
  obs = TensorDict({
    "actor": torch.randn(count, 48), "critic": torch.randn(count, 48),
  }, batch_size=[count])
  obs["actor"][:, 9:12] = 0
  obs["actor"][count // 2:, 9] = .25
  kwargs = {
    "hidden_dims": (16, 8), "obs_normalization": True,
    "distribution_cfg": {
      "class_name": "GaussianDistribution", "init_std": .17,
      "std_range": (.1, 1.0),
    },
  }
  groups = {"actor": ["actor"], "critic": ["critic"]}
  base = MLPModel(obs, groups, "actor", 12, **deepcopy(kwargs))
  quiet = QuietStopMLPModel(obs, groups, "actor", 12, **deepcopy(kwargs))
  quiet.load_state_dict(base.state_dict(), strict=True)
  return obs, base, quiet


def test_strict_weights_mean_inference_and_jit_export_are_unchanged():
  obs, base, quiet = models()
  assert base.state_dict().keys() == quiet.state_dict().keys()
  with torch.inference_mode():
    torch.testing.assert_close(base(obs), quiet(obs), rtol=0, atol=0)
    old_jit = torch.jit.script(base.as_jit())
    new_jit = torch.jit.script(quiet.as_jit())
    torch.testing.assert_close(
      old_jit(obs["actor"]), new_jit(obs["actor"]), rtol=0, atol=0
    )


def test_raw_zero_command_gate_preserves_moving_variance_and_small_yaw_motion():
  obs, base, quiet = models()
  # A physical zero maps to a large nonzero normalized command here.
  quiet.obs_normalizer._mean[:, 9:12] = torch.tensor([3., -2., 4.])
  obs["actor"][1, 11] = 5e-7
  obs["actor"][2, 11] = 2e-6
  obs["actor"][3, 10] = -.03
  quiet(obs, stochastic_output=True)
  expected = torch.full((8, 12), .17)
  expected[:2] = .01
  torch.testing.assert_close(quiet.output_std, expected)
  base(obs, stochastic_output=True)
  torch.testing.assert_close(quiet.output_std[4:], base.output_std[4:])


def test_rollout_likelihood_recomputes_exactly_and_entropy_kl_match_normal():
  obs, _, quiet = models()
  actions = quiet(obs, stochastic_output=True)
  old = tuple(value.detach().clone() for value in quiet.output_distribution_params)
  old_logprob = quiet.get_output_log_prob(actions).detach().clone()
  expected = Normal(*old)
  torch.testing.assert_close(old_logprob, expected.log_prob(actions).sum(-1))
  torch.testing.assert_close(quiet.output_entropy, expected.entropy().sum(-1))
  quiet(obs, stochastic_output=True)
  torch.testing.assert_close(quiet.get_output_log_prob(actions), old_logprob)
  with torch.no_grad():
    quiet.mlp[-1].bias.add_(.002)
  quiet(obs, stochastic_output=True)
  new = quiet.output_distribution_params
  torch.testing.assert_close(
    quiet.get_kl_divergence(old, new),
    kl_divergence(Normal(*old), Normal(*new)).sum(-1),
  )
  assert torch.isfinite(quiet.get_kl_divergence(old, new)).all()


def test_quiet_entropy_does_not_update_moving_std_and_moving_entropy_does():
  obs, _, quiet = models()
  obs["actor"][:, 9:12] = 0
  quiet(obs, stochastic_output=True)
  quiet.output_entropy.mean().backward()
  torch.testing.assert_close(
    quiet.distribution.std_param.grad, torch.zeros(12)
  )
  quiet.zero_grad(set_to_none=True)
  obs["actor"][:, 9] = .25
  quiet(obs, stochastic_output=True)
  quiet.output_entropy.mean().backward()
  assert (quiet.distribution.std_param.grad > 0).all()


def test_mirrored_minibatch_keeps_conditional_variance_and_actual_ppo_updates():
  obs, _, quiet = models(16)
  mirrored, _ = augment_blind_symmetry(None, obs)
  quiet(mirrored, stochastic_output=True)
  torch.testing.assert_close(quiet.output_std[:16], quiet.output_std[16:])
  critic = MLPModel(
    obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=(16, 8),
  )
  storage = RolloutStorage("rl", 16, 4, obs, [12], "cpu")
  algorithm = PPO(
    quiet, critic, storage, num_learning_epochs=1, num_mini_batches=2,
    symmetry_cfg={
      "use_data_augmentation": True, "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0, "data_augmentation_func": augment_blind_symmetry,
      "env": None,
    },
  )
  before = quiet.mlp[0].weight.detach().clone()
  with torch.inference_mode():
    for _ in range(4):
      action = algorithm.act(obs)
      algorithm.process_env_step(
        obs, 1 - action.square().mean(-1), torch.zeros(16), {},
      )
    algorithm.compute_returns(obs)
  losses = algorithm.update()
  assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
  assert not torch.equal(quiet.mlp[0].weight, before)


@pytest.mark.parametrize("width", [47, 49, 323])
def test_actor_rejects_other_observation_widths(width):
  obs = TensorDict({"actor": torch.zeros(2, width)}, batch_size=[2])
  with pytest.raises(ValueError, match="48D"):
    QuietStopMLPModel(obs, {"actor": ["actor"]}, "actor", 12)


def test_actor_rejects_concatenated_groups_and_non_gaussian_models():
  obs, _, _ = models()
  with pytest.raises(ValueError, match="actor-only"):
    QuietStopMLPModel(obs, {"actor": ["critic"]}, "actor", 12)
  with pytest.raises(ValueError, match="Gaussian"):
    QuietStopMLPModel(obs, {"actor": ["actor"]}, "actor", 12)


@pytest.mark.parametrize("load_cfg", [None, {"optimizer": True}, {"iteration": True}])
@pytest.mark.parametrize("identity", [None, {"standing_action_std": .17}])
def test_stateful_resume_rejects_missing_or_different_exploration(
  tmp_path, load_cfg, identity,
):
  _, _, actor = models()
  runner = LPACRLQuietStopOnPolicyRunner.__new__(LPACRLQuietStopOnPolicyRunner)
  runner.alg = SimpleNamespace(_raw_actor=actor)
  path = tmp_path / "checkpoint.pt"
  torch.save({"infos": {"lp_acrl_training_exploration": identity}}, path)
  with pytest.raises(ValueError, match="exploration differs"):
    runner.load(str(path), load_cfg=load_cfg)


def test_runner_saves_exploration_and_allows_strict_weight_only_parent_load(
  tmp_path, monkeypatch,
):
  _, _, actor = models()
  runner = LPACRLQuietStopOnPolicyRunner.__new__(LPACRLQuietStopOnPolicyRunner)
  runner.alg = SimpleNamespace(_raw_actor=actor)
  captured = {}
  monkeypatch.setattr(
    LPACRLOnPolicyRunner, "save", lambda self, path, infos: captured.update(infos)
  )
  runner.save("unused", {"existing": 5})
  assert captured["existing"] == 5
  assert captured["lp_acrl_training_exploration"] == actor.exploration_identity()
  monkeypatch.setattr(LPACRLOnPolicyRunner, "load", lambda *a, **kw: {"loaded": True})
  # Model-only initialization/inference still permits the unchanged v1 mean.
  assert runner.load("unused", load_cfg={"actor": True}) == {"loaded": True}
  path = tmp_path / "checkpoint.pt"
  torch.save({"infos": captured}, path)
  assert runner.load(str(path)) == {"loaded": True}
