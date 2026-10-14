"""Batch-consistent native exploration, finite checkpoints and correct resumes."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from test_lp_acrl_contact_runner import fixture as contact_fixture
from torch.distributions import Normal

from pochi_rl.mjlab import lp_acrl_annealed as annealed
from pochi_rl.mjlab.lp_acrl_quiet_stop import QuietStopMLPModel
from pochi_rl.mjlab.lp_acrl_resume import _equal_tree


def fixture(tmp_path, monkeypatch):
  base, saved, path, _ = contact_fixture(tmp_path, monkeypatch)
  runner = annealed.LPACRLAnnealedOnPolicyRunner.__new__(
    annealed.LPACRLAnnealedOnPolicyRunner
  )
  runner.__dict__.update(base.__dict__)
  runner.cfg = {"seed": 42, "upload_model": False}
  runner.logger = SimpleNamespace(save_model=lambda *_: None)
  runner.env.unwrapped.common_step_counter = 24
  runner.env.unwrapped.cfg = SimpleNamespace(
    observations={
      name: SimpleNamespace(enable_corruption=False) for name in ("actor", "critic")
    }
  )
  algorithm = annealed.LPACRLAnnealedPPO.__new__(annealed.LPACRLAnnealedPPO)
  algorithm.__dict__.update(base.alg.__dict__)
  algorithm.__dict__.pop("load")
  algorithm.completed_updates = 0
  algorithm.training_resume_allowed = True
  algorithm._failed_update = False
  algorithm.symmetry.use_data_augmentation = False
  distribution = algorithm._raw_actor.distribution
  distribution.std_range = [0.03, 0.35]
  distribution.std_param.requires_grad_(False)
  distribution.std_param.grad = None
  distribution.std_param.fill_(0.35)
  runner.alg = algorithm
  saved["actor_state_dict"]["distribution.std_param"].fill_(
    annealed.moving_std_for_updates(101)
  )
  saved[annealed.SCHEDULE_KEY] = {
    "identity": annealed.schedule_identity(),
    "completed_updates": 101,
    "next_moving_std": annealed.moving_std_for_updates(101),
    "training_resume_allowed": True,
  }
  saved[annealed.PPO_STATE_KEY] = {"learning_rate": 3.375e-5}
  saved["infos"].update(
    {
      "lp_acrl_training_exploration": runner.exploration_identity(),
      "lp_acrl_training_symmetry": runner.symmetry_identity(),
      "lp_acrl_observation_corruption": runner.observation_corruption_identity(),
    }
  )
  torch.save(saved, path)
  return runner, saved, path


def actual_ppo():
  obs = TensorDict({"actor": torch.zeros(4, 48)}, batch_size=[4])
  obs["actor"][2:, 9] = 0.25
  actor = QuietStopMLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    12,
    hidden_dims=(8,),
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.35,
      "learn_std": False,
      "std_range": (0.03, 0.35),
    },
  )
  critic = MLPModel(
    obs,
    {"critic": ["actor"]},
    "critic",
    1,
    hidden_dims=(8,),
    obs_normalization=True,
  )
  storage = RolloutStorage("rl", 4, 4, obs, (12,))
  return annealed.LPACRLAnnealedPPO(
    actor,
    critic,
    storage,
    num_learning_epochs=2,
    num_mini_batches=2,
  ), obs


def test_schedule_endpoints_and_monotonicity():
  values = [annealed.moving_std_for_updates(k) for k in range(602)]
  assert values[0] == 0.35 and values[600] == values[601] == 0.03
  assert values[300] == pytest.approx((0.35 * 0.03) ** 0.5)
  assert all(a > b for a, b in zip(values[:600], values[1:601], strict=True))
  for invalid in (-1, 1.5, True):
    with pytest.raises(ValueError):
      annealed.moving_std_for_updates(invalid)


def test_real_rollout_and_ppo_hold_sigma_until_successful_update():
  algorithm, obs = actual_ppo()
  observed_std = []
  hook = algorithm._raw_actor.register_forward_pre_hook(
    lambda model, _args: observed_std.append(model.distribution.std_param.clone())
  )
  with torch.inference_mode():
    for _ in range(4):
      algorithm.act(obs)
      algorithm.process_env_step(
        obs, torch.arange(4, dtype=torch.float32), torch.zeros(4, dtype=torch.bool), {}
      )
    algorithm.compute_returns(obs)
  losses = algorithm.update()
  hook.remove()
  assert len(observed_std) >= 8  # Actual collection and all PPO minibatches.
  for value in observed_std:
    torch.testing.assert_close(value, torch.full((12,), 0.35))
  expected_next = annealed.moving_std_for_updates(1)
  assert algorithm.completed_updates == 1
  assert losses["exploration/next_moving_std"] == expected_next
  assert algorithm.optimizer.state
  assert annealed._validate_schedule(algorithm.save()) == 1
  actor = algorithm._raw_actor
  mean = actor(obs).detach().clone()
  actions = actor(obs, stochastic_output=True)
  expected_std = torch.tensor([0.01, 0.01, expected_next, expected_next])[:, None]
  expected_std = expected_std.expand(4, 12)
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


def test_failed_update_does_not_advance_and_cannot_be_saved(monkeypatch):
  algorithm, _ = actual_ppo()

  def fail(_self):
    raise RuntimeError("partial optimizer failure")

  monkeypatch.setattr(PPO, "update", fail)
  with pytest.raises(RuntimeError, match="partial"):
    algorithm.update()
  assert algorithm.completed_updates == 0
  torch.testing.assert_close(
    algorithm._raw_actor.distribution.std_param, torch.full((12,), 0.35)
  )
  with pytest.raises(ValueError):
    algorithm.update()
  with pytest.raises(ValueError):
    algorithm.save()


@pytest.mark.parametrize("count", [0, 599, 600])
def test_schedule_boundary_changes_only_std_and_preserves_deterministic_mean(
  monkeypatch,
  count,
):
  algorithm, obs = actual_ppo()
  algorithm.completed_updates = count
  algorithm._raw_actor.distribution.std_param.fill_(
    annealed.moving_std_for_updates(count)
  )
  mean = algorithm._raw_actor(obs).detach().clone()

  def unchanged_update(instance):
    assert instance.completed_updates == count
    instance.validate_schedule_configuration()
    return {}

  monkeypatch.setattr(PPO, "update", unchanged_update)
  algorithm.update()
  assert algorithm.completed_updates == count + 1
  torch.testing.assert_close(algorithm._raw_actor(obs), mean)
  torch.testing.assert_close(
    algorithm._raw_actor.distribution.std_param,
    torch.full((12,), annealed.moving_std_for_updates(count + 1)),
  )


def test_full_resume_restores_lr_lp_and_next_schedule_then_saves_next_label(
  tmp_path,
  monkeypatch,
):
  runner, saved, path = fixture(tmp_path, monkeypatch)
  runner.load(str(path))
  assert runner.current_learning_iteration == runner.alg.completed_updates == 101
  assert runner.alg.learning_rate == 3.375e-5
  assert _equal_tree(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict())
  assert _equal_tree(
    saved["infos"]["lp_acrl"], runner.command_term.curriculum_state_dict()
  )
  assert runner.env.seed_calls == [42] and runner.env.reset_calls == [True]
  # A save immediately after loading still labels the last completed update.
  copy = tmp_path / "resaved.pt"
  runner.save(str(copy))
  assert runner.current_learning_iteration == 101
  assert (
    annealed.validate_annealed_checkpoint(torch.load(copy, weights_only=False)) == 101
  )
  monkeypatch.setattr(PPO, "update", lambda _self: {})
  runner.alg.update()
  runner.save(str(copy))
  result = torch.load(copy, weights_only=False)
  assert result["iter"] == 101
  assert annealed.validate_annealed_checkpoint(result) == 102


def test_actor_only_keeps_mean_but_cannot_resume_training(tmp_path, monkeypatch):
  runner, saved, path = fixture(tmp_path, monkeypatch)
  runner.load(str(path), {"actor": True})
  assert _equal_tree(saved["actor_state_dict"], runner.alg._raw_actor.state_dict())
  assert runner.alg.completed_updates == 101
  assert not runner.env.reset_calls
  with pytest.raises(ValueError, match="inference"):
    runner.learn(1)
  with pytest.raises(ValueError):
    runner.alg.update()
  copy = tmp_path / "inference.pt"
  runner.save(str(copy))
  assert (
    annealed.validate_annealed_checkpoint(torch.load(copy, weights_only=False)) == 101
  )
  with pytest.raises(ValueError, match="actor-only"):
    runner.load(str(copy))


@pytest.mark.parametrize("actor_only", [False, True])
@pytest.mark.parametrize(
  "change",
  [
    "schedule_missing",
    "count",
    "curve",
    "std",
    "next_std",
    "iteration",
    "corruption",
    "symmetry",
    "exploration",
    "learning_rate",
  ],
)
def test_bad_saved_contract_rejects_before_mutation(
  tmp_path,
  monkeypatch,
  actor_only,
  change,
):
  runner, saved, path = fixture(tmp_path, monkeypatch)
  if change == "schedule_missing":
    saved.pop(annealed.SCHEDULE_KEY)
  elif change == "count":
    saved[annealed.SCHEDULE_KEY]["completed_updates"] = True
  elif change == "curve":
    saved[annealed.SCHEDULE_KEY]["identity"]["curve"] = "linear"
  elif change == "std":
    saved["actor_state_dict"]["distribution.std_param"].fill_(0.03)
  elif change == "next_std":
    saved[annealed.SCHEDULE_KEY]["next_moving_std"] = 0.03
  elif change == "iteration":
    saved["iter"] = 101
  elif change == "corruption":
    saved["infos"]["lp_acrl_observation_corruption"]["groups"]["actor"] = True
  elif change == "symmetry":
    saved["infos"]["lp_acrl_training_symmetry"]["use_data_augmentation"] = True
  elif change == "exploration":
    saved["infos"]["lp_acrl_training_exploration"]["moving_std_learned"] = True
  else:
    saved[annealed.PPO_STATE_KEY]["learning_rate"] = 0.001
  torch.save(saved, path)
  before = deepcopy((runner.alg.save(), runner.command_term.curriculum_state_dict()))
  with pytest.raises(ValueError):
    runner.load(str(path), {"actor": True} if actor_only else None)
  assert _equal_tree(
    before, (runner.alg.save(), runner.command_term.curriculum_state_dict())
  )


@pytest.mark.parametrize("change", ["groups", "shape", "nan", "negative", "missing"])
def test_invalid_optimizer_cannot_partially_replace_networks(
  tmp_path, monkeypatch, change
):
  runner, saved, path = fixture(tmp_path, monkeypatch)
  optimizer = saved["optimizer_state_dict"]
  state = next(iter(optimizer["state"].values()))
  if change == "groups":
    optimizer["param_groups"][0]["params"].pop()
  elif change == "shape":
    state["exp_avg"] = torch.zeros(321)
  elif change == "nan":
    state["exp_avg"].fill_(float("nan"))
  elif change == "negative":
    state["exp_avg_sq"].fill_(-1)
  else:
    state.pop("step")
  torch.save(saved, path)
  before = deepcopy((runner.alg.save(), runner.command_term.curriculum_state_dict()))
  with pytest.raises(ValueError):
    runner.load(str(path))
  assert _equal_tree(
    before, (runner.alg.save(), runner.command_term.curriculum_state_dict())
  )
