"""Full resume restores adaptive PPO and partial LP while resetting simulation."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from pochi_rl.lp_acrl.curriculum import LPACRLCurriculum
from pochi_rl.mjlab.lp_acrl_quiet_stop import (
  LPACRLQuietStopOnPolicyRunner,
  QuietStopMLPModel,
)
from pochi_rl.mjlab.lp_acrl_resume import (
  TASK_ID,
  _equal_tree,
  resume_training_state,
  sha256,
  validate_resume_sources,
)
from pochi_rl.mjlab.lp_acrl_symmetry import augment_blind_symmetry


def fixture(tmp_path):
  obs = TensorDict({"actor": torch.zeros(2, 48)}, batch_size=[2])
  actor = QuietStopMLPModel(
    obs, {"actor": ["actor"]}, "actor", 12, hidden_dims=(8,),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": .15},
  )
  critic = MLPModel(
    obs, {"critic": ["actor"]}, "critic", 1, hidden_dims=(8,),
    obs_normalization=True,
  )
  optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=.001)
  loss = actor(obs).sum() + critic(obs).sum() + actor.distribution.std_param.sum()
  loss.backward()
  optimizer.step()
  optimizer.param_groups[0]["lr"] = 3.375e-5
  symmetry = SimpleNamespace(
    data_augmentation_func=augment_blind_symmetry,
    use_data_augmentation=True, use_mirror_loss=True, mirror_loss_coeff=1.,
  )
  algorithm = SimpleNamespace(
    _raw_actor=actor, _raw_critic=critic, optimizer=optimizer, rnd=None,
    symmetry=symmetry, learning_rate=.001,
  )
  algorithm.load = lambda saved, cfg, strict: PPO.load(algorithm, saved, cfg, strict)
  identity = {"environment": {"physics": .0025, "rewards": {"duty": -8}},
              "command_protocol": {"standing": .2}}
  curriculum = LPACRLCurriculum(600)
  term = SimpleNamespace(
    _env=SimpleNamespace(cfg=SimpleNamespace(scene=SimpleNamespace(
      terrain=SimpleNamespace(terrain_generator=SimpleNamespace(seed=42))
    ))), curriculum=curriculum, checkpoint_identity=lambda: identity,
  )
  term.curriculum_state_dict = lambda: {
    "task_space": deepcopy(identity), "training_terrain_seed": 42,
    "curriculum": curriculum.state_dict(),
  }
  term.load_curriculum_state_dict = lambda saved: curriculum.load_state_dict(
    saved["curriculum"]
  )
  environment = SimpleNamespace(
    unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term)),
    episode_length_buf=torch.zeros(2, dtype=torch.long), reset_calls=[], seed_calls=[],
  )
  environment.seed = environment.seed_calls.append
  environment.reset = lambda: environment.reset_calls.append(True)
  runner = LPACRLQuietStopOnPolicyRunner.__new__(LPACRLQuietStopOnPolicyRunner)
  runner.env, runner.alg, runner.device = environment, algorithm, "cpu"
  runner.current_learning_iteration = 0
  curriculum.record_episodes(torch.arange(600).repeat(2), torch.ones(1200))
  curriculum.end_stage()
  curriculum.record_episodes(torch.tensor([5]), torch.tensor([2.]))
  curriculum.advance(24)
  saved = deepcopy({
    "iter": 100, "actor_state_dict": actor.state_dict(),
    "critic_state_dict": critic.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "infos": {
      "lp_acrl_checkpoint_version": 1, "lp_acrl": term.curriculum_state_dict(),
      "lp_acrl_training_symmetry": runner.symmetry_identity(),
      "lp_acrl_training_exploration": runner.exploration_identity(),
    },
  })
  # Leave a fresh target runner, with visibly different weights and LR.
  with torch.no_grad():
    for model in (actor, critic):
      for value in model.parameters():
        value.add_(1.)
  optimizer.state.clear()
  optimizer.param_groups[0]["lr"] = .001
  curriculum.load_state_dict(LPACRLCurriculum(600).state_dict())
  path = tmp_path / "model_100.pt"
  torch.save(saved, path)
  return runner, saved, path, {"task": TASK_ID, "seed": 42}


def test_actual_rsl_full_load_restores_adaptive_lr_optimizer_and_partial_lp(tmp_path):
  runner, saved, path, parent = fixture(tmp_path)
  result = resume_training_state(runner, path, parent, 42)
  assert runner.current_learning_iteration == result["next_iteration"] == 101
  assert runner.alg.learning_rate == 3.375e-5
  assert runner.alg.optimizer.param_groups[0]["lr"] == 3.375e-5
  assert _equal_tree(saved["optimizer_state_dict"], runner.alg.optimizer.state_dict())
  assert _equal_tree(
    saved["infos"]["lp_acrl"], runner.command_term.curriculum_state_dict()
  )
  assert result["lp_stage"] == 1 and result["lp_steps_in_stage"] == 24
  assert runner.env.seed_calls == [42] and runner.env.reset_calls == [True]
  assert result["simulator_state_and_rng_restored"] is False


@pytest.mark.parametrize("change", [
  "reward", "physics", "command", "terrain", "exploration", "symmetry",
])
def test_all_training_identity_changes_are_rejected_before_mutation(tmp_path, change):
  runner, saved, path, parent = fixture(tmp_path)
  before = deepcopy(runner.alg._raw_actor.state_dict())
  metadata = saved["infos"]
  if change in ("reward", "physics"):
    field = "physics" if change == "physics" else "rewards"
    metadata["lp_acrl"]["task_space"]["environment"][field] = "changed"
  elif change == "command":
    metadata["lp_acrl"]["task_space"]["command_protocol"] = {"changed": True}
  elif change == "terrain":
    metadata["lp_acrl"]["training_terrain_seed"] = 99
  elif change == "exploration":
    metadata["lp_acrl_training_exploration"]["standing_action_std"] = .03
  else:
    metadata["lp_acrl_training_symmetry"]["mirror_loss_coeff"] = 0.
  torch.save(saved, path)
  with pytest.raises(ValueError):
    resume_training_state(runner, path, parent, 42)
  assert _equal_tree(before, runner.alg._raw_actor.state_dict())


@pytest.mark.parametrize("change", [
  "weights", "optimizer", "lr", "iteration", "family", "seed",
])
def test_invalid_state_and_wrong_lineage_are_rejected(tmp_path, change):
  runner, saved, path, parent = fixture(tmp_path)
  if change == "weights":
    saved["critic_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
  elif change == "optimizer":
    next(iter(saved["optimizer_state_dict"]["state"].values()))["exp_avg"].fill_(float("nan"))
  elif change == "lr":
    saved["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.
  elif change == "iteration":
    saved["iter"] = -1
  elif change == "family":
    parent["task"] = "Pochi-LP-ACRL-Blind-v2"
  else:
    parent["seed"] = 123
  torch.save(saved, path)
  with pytest.raises(ValueError):
    resume_training_state(runner, path, parent, 42)


def test_reset_must_not_add_partial_episode_lp_credit(tmp_path):
  runner, _, path, parent = fixture(tmp_path)
  runner.env.reset = lambda: runner.command_term.curriculum.record_episodes(
    torch.tensor([0]), torch.tensor([1.])
  )
  with pytest.raises(ValueError, match="reset altered"):
    resume_training_state(runner, path, parent, 42)


def test_all_recorded_sources_and_assets_must_match(tmp_path):
  names = [
    "assets/pochi/pochi.xml", "assets/pochi/meshes/foot.obj",
    "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py",
    "src/pochi_rl/mjlab/lp_acrl_blind_v3_rewards.py",
    "src/pochi_rl/mjlab/lp_acrl_runner.py", "src/other.py",
  ]
  expected = {}
  for name in names:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    expected[name] = sha256(path)
  parent = {"task": TASK_ID, "source_sha256": expected}
  assert validate_resume_sources(tmp_path, parent) == expected
  (tmp_path / "src/other.py").write_text("changed")
  with pytest.raises(ValueError, match="source or physical asset"):
    validate_resume_sources(tmp_path, parent)
