"""Paper equations, stage accounting, catalog coverage, and resume integrity."""

import copy

import pytest
import torch

from pochi_rl.lp_acrl import (
  FLAT_TASK_CATALOG,
  LP_ACRL_SPEC,
  ROUGH_TASK_CATALOG,
  LPACRLCurriculum,
  TaskCatalog,
  epte_sp,
  percentage_tracking_error,
)


def record(curriculum, ids, returns):
  curriculum.record_episodes(
    torch.tensor(ids), torch.tensor(returns, dtype=torch.float64)
  )


def test_signed_progress_and_paper_softmax():
  curriculum = LPACRLCurriculum(3, beta=2.0, min_episodes=2)
  record(curriculum, [0, 1, 2, 0, 1, 2], [8.0, 20, 28, 12, 20, 32])
  first = curriculum.end_stage()
  torch.testing.assert_close(
    first.mean_returns, torch.tensor([10.0, 20.0, 30.0]).double()
  )
  torch.testing.assert_close(first.probabilities, torch.full((3,), 1 / 3).double())
  assert not first.compared_tasks.any()
  record(curriculum, [0, 0, 1, 1, 2, 2], [12.0, 16, 20, 20, 24, 28])
  second = curriculum.end_stage()
  torch.testing.assert_close(
    second.learning_progress, torch.tensor([4.0, 0.0, -4.0]).double()
  )
  torch.testing.assert_close(
    second.probabilities, torch.softmax(torch.tensor([2.0, 0.0, -2.0]).double(), 0)
  )
  assert second.probabilities[0] > second.probabilities[1] > second.probabilities[2]
  assert second.compared_tasks.all()
  assert first.episode_counts.tolist() == [2, 2, 2]
  assert curriculum.episode_counts.tolist() == [0, 0, 0]


def test_absent_and_new_tasks_do_not_create_spurious_progress():
  curriculum = LPACRLCurriculum(3, min_episodes=2)
  record(curriculum, [0, 0, 1], [5.0, 5, 50])
  curriculum.end_stage()
  record(curriculum, [1, 1, 2, 2], [100.0, 100, 1000, 1000])
  stage = curriculum.end_stage()
  assert stage.learning_progress.tolist() == [0.0, 0.0, 0.0]
  record(curriculum, [0, 0, 1, 1, 2], [500.0, 500, 105, 105, 2000])
  stage = curriculum.end_stage()
  assert stage.learning_progress.tolist() == [0.0, 5.0, 0.0]
  assert stage.compared_tasks.tolist() == [False, True, False]


def test_stage_counts_control_steps_and_uses_only_completed_episodes():
  curriculum = LPACRLCurriculum(2, stage_steps=24, min_episodes=1)
  assert curriculum.advance(23) is None
  record(curriculum, [1], [12.0])
  stage = curriculum.advance()
  assert stage.stage_index == 1
  assert stage.episode_counts.tolist() == [0, 1]
  assert curriculum.steps_in_stage == 0
  assert curriculum.advance(25).stage_index == 2
  assert curriculum.steps_in_stage == 1
  with pytest.raises(ValueError, match="more than one"):
    curriculum.advance(48)
  assert curriculum.steps_in_stage == 1


def test_extreme_finite_returns_and_tiny_beta_still_sample_finitely():
  curriculum = LPACRLCurriculum(2, beta=1e-300, min_episodes=1)
  record(curriculum, [0, 1], [-1e300, 1e300])
  curriculum.end_stage()
  record(curriculum, [0, 1], [1e300, -1e300])
  curriculum.end_stage()
  assert torch.isfinite(curriculum.probabilities).all()
  assert curriculum.probabilities.tolist() == [1.0, 0.0]
  assert curriculum.sample(20).tolist() == [0] * 20
  assert curriculum.sample(0).shape == (0,)


def test_sampling_matches_distribution_and_generator_is_reproducible():
  curriculum = LPACRLCurriculum(3, beta=1.0, min_episodes=1)
  record(curriculum, [0, 1, 2], [0.0, 0, 0])
  curriculum.end_stage()
  record(curriculum, [0, 1, 2], [1.0, 0, -1])
  curriculum.end_stage()
  first = curriculum.sample(20_000, generator=torch.Generator().manual_seed(37))
  second = curriculum.sample(20_000, generator=torch.Generator().manual_seed(37))
  assert torch.equal(first, second)
  frequencies = torch.bincount(first, minlength=3).double() / first.numel()
  torch.testing.assert_close(frequencies, curriculum.probabilities, atol=0.01, rtol=0)


def test_resume_preserves_partial_stage_and_next_update():
  original = LPACRLCurriculum(2, min_episodes=1, stage_steps=10)
  record(original, [0, 1], [10.0, 20])
  original.advance(10)
  record(original, [0], [12.0])
  original.advance(4)
  state = original.state_dict()
  resumed = LPACRLCurriculum(2, min_episodes=1, stage_steps=10)
  resumed.load_state_dict(state)
  state["return_sums"].fill_(-999)  # No mutable checkpoint aliasing.
  for curriculum in (original, resumed):
    record(curriculum, [1], [17.0])
    curriculum.advance(6)
  torch.testing.assert_close(original.probabilities, resumed.probabilities)
  assert original.stage_index == resumed.stage_index == 2
  assert resumed.learning_progress.tolist() == [2.0, -3.0]


@pytest.mark.parametrize(
  ("key", "bad_value"),
  [
    ("beta", 999.0),
    ("num_tasks", 4),
    ("steps_in_stage", 2400),
    ("stage_index", -1),
    ("previous_means", torch.tensor([float("nan"), 0.0]).double()),
    ("episode_counts", torch.tensor([-1, 0])),
    ("probabilities", torch.tensor([0.2, 0.2]).double()),
    ("probabilities", torch.tensor([0.2, 0.8]).double()),
    ("previous_valid", torch.zeros(2)),
    ("return_sums", torch.ones(2).double()),
  ],
)
def test_bad_resume_is_rejected_atomically(key, bad_value):
  curriculum = LPACRLCurriculum(2)
  initial = curriculum.state_dict()
  bad = copy.deepcopy(initial)
  bad[key] = bad_value
  with pytest.raises(ValueError):
    curriculum.load_state_dict(bad)
  restored = curriculum.state_dict()
  for name, value in initial.items():
    if isinstance(value, torch.Tensor):
      assert torch.equal(value, restored[name])
    else:
      assert value == restored[name]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(bad):
  with pytest.raises(ValueError, match="beta"):
    LPACRLCurriculum(2, beta=bad)


@pytest.mark.parametrize(
  ("ids", "returns"),
  [
    (torch.tensor([2]), torch.tensor([1.0])),
    (torch.tensor([-1]), torch.tensor([1.0])),
    (torch.tensor([0.0]), torch.tensor([1.0])),
    (torch.tensor([0]), torch.tensor([float("nan")])),
    (torch.tensor([0, 1]), torch.tensor([1.0])),
  ],
)
def test_invalid_completed_episodes_do_not_modify_state(ids, returns):
  curriculum = LPACRLCurriculum(2)
  with pytest.raises(ValueError):
    curriculum.record_episodes(ids, returns)
  assert not curriculum.episode_counts.any()
  assert not curriculum.return_sums.any()


def test_catalog_covers_complete_cartesian_product_and_flat_is_forward_only():
  catalog = ROUGH_TASK_CATALOG
  indices = catalog.task_indices()
  assert indices.shape == (600, 4)
  assert indices.unique(dim=0).shape[0] == 600
  assert indices[0].tolist() == [0, 0, 0, 0]
  assert indices[-1].tolist() == [4, 5, 4, 3]
  for axis, count in enumerate((5, 6, 5, 4)):
    assert torch.bincount(indices[:, axis]).tolist() == [600 // count] * count
  bounds = catalog.task_bounds()
  torch.testing.assert_close(bounds[0], torch.tensor([[0.0, 0.1], [0.0, 0.2]]))
  torch.testing.assert_close(bounds[-1], torch.tensor([[0.4, 0.5], [1.0, 1.2]]))
  assert FLAT_TASK_CATALOG.num_tasks == 5
  assert not FLAT_TASK_CATALOG.task_bounds()[:, 1].any()
  assert LP_ACRL_SPEC.episode_steps == 1000
  assert LP_ACRL_SPEC.stage_steps == 100 * LP_ACRL_SPEC.ppo_steps_per_iteration


def test_invalid_task_edges_are_rejected():
  with pytest.raises(ValueError, match="strictly increasing"):
    TaskCatalog((0.0, 0.1, 0.1), (0.0, 0.0), ("flat",), 1)
  with pytest.raises(ValueError, match="finite"):
    TaskCatalog((0.0, float("nan")), (0.0, 0.0), ("flat",), 1)


def test_percentage_error_has_explicit_zero_command_convention():
  actual = torch.tensor([0.15, -0.15, 0.0, 0.025, 0.2])
  command = torch.tensor([0.1, -0.1, 0.0, 0.0, 0.0])
  result = percentage_tracking_error(actual, command, denominator_floor=0.05)
  torch.testing.assert_close(result, torch.tensor([0.5, 0.5, 0.0, 0.5, 4.0]))


def test_epte_sp_equation_and_per_axis_survival_penalty():
  errors = torch.tensor([[0.2, 0.4], [0.2, 0.4], [0.0, 0.0]])
  counts = torch.tensor([[1000], [500], [0]])
  result = epte_sp(errors, counts, 1000)
  torch.testing.assert_close(result, torch.tensor([[0.2, 0.4], [0.6, 0.7], [1.0, 1.0]]))
  assert epte_sp(torch.tensor(1.5), 1000, 1000).item() == 1.5


@pytest.mark.parametrize("count", [-1, 1001, 2.5, float("nan")])
def test_epte_rejects_invalid_survival_count(count):
  with pytest.raises(ValueError, match="alive_steps"):
    epte_sp(torch.tensor(0.2), count, 1000)
