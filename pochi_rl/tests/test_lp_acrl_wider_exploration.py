"""Full-state variance migration and runner boundaries with actual CPU PPO."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from test_lp_acrl_contact_runner import fixture as contact_fixture
from torch.distributions import Normal

from pochi_rl.mjlab import lp_acrl_wider_exploration as wider
from pochi_rl.mjlab.lp_acrl_low_noise import expected_exploration_identity
from pochi_rl.mjlab.lp_acrl_resume import _equal_tree
from pochi_rl.mjlab.lp_acrl_symmetry import augment_blind_symmetry


def fixture(tmp_path, monkeypatch):
  tmp_path.mkdir(parents=True, exist_ok=True)
  base, saved, path, backend = contact_fixture(tmp_path, monkeypatch)
  runner = wider.LPACRLWiderExplorationOnPolicyRunner.__new__(
    wider.LPACRLWiderExplorationOnPolicyRunner
  )
  runner.__dict__.update(base.__dict__)
  runner._training_ready = runner._in_learning = runner._failed_learning = False
  runner.cfg = {
    "seed": 42,
    "upload_model": False,
    "num_steps_per_env": 4,
    "save_interval": 1,
    "algorithm": {"rnd_cfg": None},
  }
  runner.is_distributed = False
  runner.env.device = "cpu"
  runner.env.max_episode_length = 1000
  runner.env.unwrapped.common_step_counter = 0
  runner.env.unwrapped.cfg = SimpleNamespace(
    observations={
      name: SimpleNamespace(enable_corruption=False) for name in ("actor", "critic")
    }
  )
  obs = TensorDict({"actor": torch.zeros(2, 48)}, batch_size=[2])
  obs["actor"][1, 9] = 0.25
  runner.env.get_observations = lambda: obs.clone()
  actor, critic = base.alg._raw_actor, base.alg._raw_critic
  actor.distribution.std_range = (0.15, 0.15)
  actor.distribution.std_param.requires_grad_(False)
  actor.distribution.std_param.grad = None
  actor.distribution.std_param.fill_(0.15)
  runner.alg = PPO(
    actor,
    critic,
    RolloutStorage("rl", 2, 4, obs, (12,)),
    num_learning_epochs=2,
    num_mini_batches=2,
    symmetry_cfg={
      "env": runner.env,
      "data_augmentation_func": augment_blind_symmetry,
      "use_data_augmentation": False,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0,
    },
  )
  saved["iter"] = 300
  saved["actor_state_dict"][wider.STD_PARAMETER].fill_(0.03)
  saved["infos"].update(
    {
      "env_state": {"common_step_counter": 7224},
      "lp_acrl_observation_corruption": deepcopy(wider.CORRUPTION_IDENTITY),
      "lp_acrl_training_exploration": expected_exploration_identity(),
      "lp_acrl_training_symmetry": wider.expected_symmetry_identity(),
    }
  )
  # Nontrivial normalizer buffers must survive as well as mean-network weights.
  for key in ("actor_state_dict", "critic_state_dict"):
    for name, value in saved[key].items():
      if name.startswith("obs_normalizer.") and value.is_floating_point():
        value.add_(0.125)
  runner.logged_iterations = []
  runner.logger = SimpleNamespace(
    writer=object(),
    log_dir=str(tmp_path),
    init_logging_writer=lambda: None,
    stop_logging_writer=lambda: None,
    process_env_step=lambda *_: None,
    save_model=lambda *_: None,
    log=lambda **kw: runner.logged_iterations.append(kw["it"]),
  )

  def step(actions):
    assert actions.shape == (2, 12) and torch.isfinite(actions).all()
    runner.env.unwrapped.common_step_counter += 1
    runner.env.episode_length_buf += 1
    runner.command_term.curriculum.advance()
    return obs.clone(), torch.tensor([0.1, 0.3]), torch.zeros(2, dtype=torch.bool), {}

  runner.env.step = step
  torch.save(saved, path)
  parent = {"task": wider.SOURCE_TASK, "seed": 42, "contact_backend": backend}
  return runner, saved, path, parent


def snapshot(runner):
  return deepcopy(
    (
      runner.alg.save(),
      runner.command_term.curriculum_state_dict(),
      runner.alg.learning_rate,
      runner.current_learning_iteration,
      runner.env.unwrapped.common_step_counter,
    )
  )


def initialize(runner, saved, parent):
  return wider.initialize_wider_exploration(runner, saved, parent)


def migrated_checkpoint(tmp_path, monkeypatch):
  runner, saved, _, parent = fixture(tmp_path, monkeypatch)
  initialize(runner, saved, parent)
  path = tmp_path / "initial_state.pt"
  runner.save(str(path), {"caller": "preserved"})
  return runner, torch.load(path, weights_only=False), path


def test_full_migration_changes_only_std_preserves_models_adam_lp_lr_counters(
  tmp_path,
  monkeypatch,
):
  runner, saved, _, parent = fixture(tmp_path, monkeypatch)
  immutable_source = deepcopy(saved)
  result = initialize(runner, saved, parent)
  expected_actor = deepcopy(saved["actor_state_dict"])
  expected_actor[wider.STD_PARAMETER].fill_(0.15)
  assert _equal_tree(expected_actor, runner.alg._raw_actor.state_dict())
  assert _equal_tree(saved["critic_state_dict"], runner.alg._raw_critic.state_dict())
  assert _equal_tree(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict())
  assert _equal_tree(
    saved["infos"]["lp_acrl"], runner.command_term.curriculum_state_dict()
  )
  assert _equal_tree(immutable_source, saved)
  assert runner.alg.learning_rate == 3.375e-5
  assert runner.current_learning_iteration == runner.completed_updates() == 301
  assert runner.env.unwrapped.common_step_counter == 7224
  assert runner.env.seed_calls == [42] and runner.env.reset_calls == [True]
  assert result["simulator_state_and_rng_restored"] is False
  assert runner._training_ready and not runner._failed_learning
  obs = runner.env.get_observations()
  mean = runner.alg.actor(obs).detach().clone()
  action = runner.alg.actor(obs, stochastic_output=True)
  std = torch.tensor([0.01, 0.15])[:, None].expand(2, 12)
  torch.testing.assert_close(runner.alg.actor.output_std, std)
  torch.testing.assert_close(
    runner.alg.actor.get_output_log_prob(action),
    Normal(mean, std).log_prob(action).sum(-1),
  )
  torch.testing.assert_close(runner.alg.actor(obs), mean)


def test_real_cpu_ppo_learn_save_resume_and_repeated_learn_use_next_iteration(
  tmp_path,
  monkeypatch,
):
  runner, saved, path = migrated_checkpoint(tmp_path, monkeypatch)
  assert saved["iter"] == 300
  assert wider.validate_wider_exploration_checkpoint(saved) == 301
  assert saved["infos"]["caller"] == "preserved"
  assert runner.current_learning_iteration == 301
  runner.learn(2)
  assert runner.logged_iterations == [301, 302]
  assert runner.current_learning_iteration == runner.completed_updates() == 303
  saved = torch.load(tmp_path / "model_302.pt", weights_only=False)
  assert wider.validate_wider_exploration_checkpoint(saved) == 303
  assert saved["infos"]["env_state"]["common_step_counter"] == 7232
  assert all(torch.isfinite(v).all() for v in saved["actor_state_dict"].values())
  target, _, _, _ = fixture(tmp_path / "resume", monkeypatch)
  target.load(str(tmp_path / "model_302.pt"))
  assert target.current_learning_iteration == 303
  assert _equal_tree(saved["optimizer_state_dict"], target.alg.optimizer.state_dict())
  assert target.alg.learning_rate == saved["infos"][wider.STATE_KEY]["learning_rate"]
  assert target.env.unwrapped.common_step_counter == 7232
  target.learn(1)
  target.learn(1)
  assert target.logged_iterations == [303, 304]
  assert target.current_learning_iteration == target.completed_updates() == 305
  final = torch.load(tmp_path / "resume/model_304.pt", weights_only=False)
  assert wider.validate_wider_exploration_checkpoint(final) == 305
  assert final["infos"]["lp_acrl"]["curriculum"]["steps_in_stage"] == 40
  assert final["infos"]["env_state"]["common_step_counter"] == 7240
  assert path.exists()


def test_actor_only_load_restores_mean_without_optimizer_lp_or_training(
  tmp_path,
  monkeypatch,
):
  _, saved, path = migrated_checkpoint(tmp_path, monkeypatch)
  runner, _, _, _ = fixture(tmp_path / "actor", monkeypatch)
  before = deepcopy(
    (
      runner.alg._raw_critic.state_dict(),
      runner.alg.optimizer.state_dict(),
      runner.command_term.curriculum_state_dict(),
    )
  )
  runner.load(str(path), {"actor": True})
  assert _equal_tree(saved["actor_state_dict"], runner.alg._raw_actor.state_dict())
  assert _equal_tree(
    before,
    (
      runner.alg._raw_critic.state_dict(),
      runner.alg.optimizer.state_dict(),
      runner.command_term.curriculum_state_dict(),
    ),
  )
  assert not runner.env.reset_calls
  assert torch.isfinite(runner.alg.actor(runner.env.get_observations())).all()
  with pytest.raises(ValueError, match="failed runner cannot"):
    runner.learn(1)
  with pytest.raises(ValueError, match="failed runner cannot"):
    runner.save(str(tmp_path / "inference.pt"))
  assert not (tmp_path / "inference.pt").exists()


@pytest.mark.parametrize(
  "change",
  [
    "source_task",
    "source_iter",
    "source_iter_float",
    "seed",
    "exploration",
    "corruption",
    "symmetry",
    "backend",
    "environment",
    "normalizer_nan",
    "critic_shape",
    "optimizer_groups",
    "optimizer_moment",
    "lp_shape",
    "lp_nan",
    "env_counter",
    "lr",
  ],
)
def test_bad_migration_rejects_before_model_optimizer_or_counter_mutation(
  tmp_path,
  monkeypatch,
  change,
):
  runner, saved, _, parent = fixture(tmp_path, monkeypatch)
  if change == "source_task":
    parent["task"] = "Pochi-LP-ACRL-Blind-v5"
  elif change == "source_iter":
    saved["iter"] = 299
  elif change == "source_iter_float":
    saved["iter"] = 300.0
  elif change == "seed":
    parent["seed"] = 123
  elif change in ("exploration", "corruption", "symmetry", "backend"):
    keys = {
      "exploration": "lp_acrl_training_exploration",
      "corruption": "lp_acrl_observation_corruption",
      "symmetry": "lp_acrl_training_symmetry",
      "backend": "lp_acrl_contact_backend",
    }
    saved["infos"][keys[change]] = {}
  elif change == "environment":
    saved["infos"]["lp_acrl"]["task_space"]["environment"]["physics"] = 0.1
  elif change == "normalizer_nan":
    key = next(
      k
      for k, v in saved["actor_state_dict"].items()
      if k.startswith("obs_normalizer.") and v.is_floating_point()
    )
    saved["actor_state_dict"][key].fill_(float("nan"))
  elif change == "critic_shape":
    saved["critic_state_dict"]["mlp.0.weight"] = torch.zeros(3, 3)
  elif change == "optimizer_groups":
    saved["optimizer_state_dict"]["param_groups"][0]["params"].pop()
  elif change == "optimizer_moment":
    next(iter(saved["optimizer_state_dict"]["state"].values()))["exp_avg_sq"].fill_(-1)
  elif change == "lp_shape":
    saved["infos"]["lp_acrl"]["curriculum"]["probabilities"] = torch.ones(4)
  elif change == "lp_nan":
    saved["infos"]["lp_acrl"]["curriculum"]["return_sums"].fill_(float("nan"))
  elif change == "env_counter":
    saved["infos"]["env_state"]["common_step_counter"] = True
  else:
    saved["optimizer_state_dict"]["param_groups"][0]["lr"] = float("inf")
  before = snapshot(runner)
  with pytest.raises(ValueError):
    initialize(runner, saved, parent)
  assert _equal_tree(before, snapshot(runner))
  assert not runner.env.reset_calls


@pytest.mark.parametrize("actor_only", [False, True])
@pytest.mark.parametrize(
  "change", ["missing_state", "count", "iter", "lr", "std", "counter"]
)
def test_bad_resume_metadata_rejected_before_mutation(
  tmp_path, monkeypatch, actor_only, change
):
  _, saved, path = migrated_checkpoint(tmp_path, monkeypatch)
  runner, _, _, _ = fixture(tmp_path / "target", monkeypatch)
  if change == "missing_state":
    saved["infos"].pop(wider.STATE_KEY)
  elif change == "count":
    saved["infos"][wider.STATE_KEY]["completed_updates"] = 302
  elif change == "iter":
    saved["iter"] = 299
  elif change == "lr":
    saved["infos"][wider.STATE_KEY]["learning_rate"] = 0.001
  elif change == "std":
    saved["actor_state_dict"][wider.STD_PARAMETER].fill_(0.03)
  else:
    saved["infos"]["env_state"]["common_step_counter"] = -1
  torch.save(saved, path)
  before = snapshot(runner)
  with pytest.raises(ValueError):
    runner.load(str(path), {"actor": True} if actor_only else None)
  assert _equal_tree(before, snapshot(runner))
  assert not runner.env.reset_calls


@pytest.mark.parametrize("stage", ["migration", "full_resume"])
@pytest.mark.parametrize("failure", ["raise", "lp_credit", "episode_age", "counter"])
def test_failed_fresh_reset_poisons_training_and_save(
  tmp_path, monkeypatch, stage, failure
):
  source, saved, path = migrated_checkpoint(tmp_path, monkeypatch)
  runner, parent_saved, _, parent = fixture(tmp_path / "target", monkeypatch)

  def reset():
    if failure == "raise":
      raise RuntimeError("reset failed")
    if failure == "lp_credit":
      runner.command_term.curriculum.record_episodes(
        torch.tensor([0]), torch.tensor([1.0])
      )
    elif failure == "episode_age":
      runner.env.episode_length_buf.fill_(1)
    else:
      runner.env.unwrapped.common_step_counter += 1

  runner.env.reset = reset
  with pytest.raises((ValueError, RuntimeError)):
    if stage == "migration":
      initialize(runner, parent_saved, parent)
    else:
      runner.load(str(path))
  assert runner._failed_learning and not runner._training_ready
  for operation in (
    lambda: runner.learn(1),
    lambda: runner.save(str(tmp_path / "bad.pt")),
  ):
    with pytest.raises(ValueError, match="failed runner cannot"):
      operation()
  assert not (tmp_path / "bad.pt").exists()
  assert source._training_ready and saved["iter"] == 300


def test_partial_ppo_failure_cannot_be_followed_by_save_or_more_learning(
  tmp_path, monkeypatch
):
  runner, _, _ = migrated_checkpoint(tmp_path, monkeypatch)

  def fail(algorithm):
    with torch.no_grad():
      next(algorithm.actor.parameters()).add_(0.1)
    raise RuntimeError("partial Adam update")

  monkeypatch.setattr(PPO, "update", fail)
  with pytest.raises(RuntimeError, match="partial Adam"):
    runner.learn(1)
  assert runner._failed_learning and not runner._in_learning
  with pytest.raises(ValueError, match="failed runner cannot"):
    runner.save(str(tmp_path / "partial.pt"))
  with pytest.raises(ValueError, match="failed runner cannot"):
    runner.learn(1)
  assert not (tmp_path / "partial.pt").exists()


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_save_rejects_invalid_adaptive_lr(tmp_path, monkeypatch, rate):
  runner, _, _ = migrated_checkpoint(tmp_path, monkeypatch)
  runner.alg.learning_rate = rate
  for group in runner.alg.optimizer.param_groups:
    group["lr"] = rate
  with pytest.raises(ValueError):
    runner.save(str(tmp_path / "bad_lr.pt"))
  assert not (tmp_path / "bad_lr.pt").exists()
