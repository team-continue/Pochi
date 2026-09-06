"""The joint frame the walk checkpoint speaks, vs the one the robot speaks.

These pin the conversion in ``run_walk_hardware`` against the thing that
actually defines it: the joint angles the training env used before 9c1f1d3.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

run_walk_hardware = pytest.importorskip("run_walk_hardware")

from pochi_rl.robot import JOINT_NAMES  # noqa: E402

ACTION_SCALE = 0.25

# `DEFAULT_JOINT_POS` as it stood at 32d6ea3, the commit
# `origin/feature/deliberate-walk` branches from -- so this, and the
# all-positive joint axes that went with it, is the frame every existing
# checkpoint was trained in. Copied out of git rather than imported, on
# purpose: the point of the test is that it is *not* the same as today's.
CANONICAL_DEFAULT_JOINT_POS = {
    "FL_hip_roll": 0.0,
    "FL_hip_pitch": 0.8,
    "FL_knee": -1.5,
    "FR_hip_roll": 0.0,
    "FR_hip_pitch": 0.8,
    "FR_knee": -1.5,
    "RL_hip_roll": 0.0,
    "RL_hip_pitch": -0.8,
    "RL_knee": 1.5,
    "RR_hip_roll": 0.0,
    "RR_hip_pitch": -0.8,
    "RR_knee": 1.5,
}


def test_canonical_sign_maps_todays_stance_onto_the_trained_one():
    canonical = np.array([CANONICAL_DEFAULT_JOINT_POS[n] for n in JOINT_NAMES])
    sign = run_walk_hardware.frame_sign("canonical")
    assert np.allclose(sign * run_walk_hardware._DEFAULT_JOINT_POS, canonical)


def test_exactly_six_joints_are_inverted():
    sign = run_walk_hardware.frame_sign("canonical")
    inverted = {name for name, s in zip(JOINT_NAMES, sign, strict=True) if s < 0}
    assert inverted == {
        "FL_hip_roll",
        "FL_hip_pitch",
        "FL_knee",
        "FR_hip_roll",
        "RL_hip_pitch",
        "RL_knee",
    }
    assert np.array_equal(run_walk_hardware.frame_sign("motor"), np.ones(12))


@pytest.mark.parametrize("frame", ["canonical", "motor"])
def test_zero_action_is_the_default_stance_in_either_frame(frame):
    sign = run_walk_hardware.frame_sign(frame)
    targets = run_walk_hardware.joint_targets(np.zeros(12), sign)
    assert np.allclose(targets, run_walk_hardware._DEFAULT_JOINT_POS)


@pytest.mark.parametrize("frame", ["canonical", "motor"])
def test_observation_and_targets_are_inverses(frame):
    """A robot that tracked the last targets perfectly must observe exactly the
    action that produced them -- the conversion has to be undone on the way in
    by the same sign it was applied with on the way out."""
    sign = run_walk_hardware.frame_sign(frame)
    action = np.linspace(-1.0, 1.0, 12)
    targets = run_walk_hardware.joint_targets(action, sign)
    obs = run_walk_hardware.build_observation(
        base_lin_vel=np.zeros(3),
        base_ang_vel=np.zeros(3),
        gravity_body=np.array([0.0, 0.0, -1.0]),
        command=np.zeros(3),
        joint_pos=targets,
        joint_vel=np.zeros(12),
        last_action=action,
        elapsed_s=0.0,
        sign=sign,
    )
    joint_pos_rel = obs[12:24]
    assert np.allclose(joint_pos_rel, ACTION_SCALE * action, atol=1e-6)


def test_joint_velocity_gets_the_same_sign_as_position():
    sign = run_walk_hardware.frame_sign("canonical")
    joint_vel = np.arange(1.0, 13.0)
    obs = run_walk_hardware.build_observation(
        base_lin_vel=np.zeros(3),
        base_ang_vel=np.zeros(3),
        gravity_body=np.array([0.0, 0.0, -1.0]),
        command=np.zeros(3),
        joint_pos=run_walk_hardware._DEFAULT_JOINT_POS,
        joint_vel=joint_vel,
        last_action=np.zeros(12),
        elapsed_s=0.0,
        sign=sign,
    )
    assert np.allclose(obs[24:36], sign * joint_vel)


def test_last_action_is_fed_back_unconverted():
    """`last_action` is the network's own output, not a joint angle."""
    sign = run_walk_hardware.frame_sign("canonical")
    last_action = np.linspace(-1.0, 1.0, 12)
    obs = run_walk_hardware.build_observation(
        base_lin_vel=np.zeros(3),
        base_ang_vel=np.zeros(3),
        gravity_body=np.array([0.0, 0.0, -1.0]),
        command=np.zeros(3),
        joint_pos=run_walk_hardware._DEFAULT_JOINT_POS,
        joint_vel=np.zeros(12),
        last_action=last_action,
        elapsed_s=0.0,
        sign=sign,
    )
    assert np.allclose(obs[36:48], last_action)


def test_observation_layout_covers_the_whole_vector():
    total = sum(dim for _, dim in run_walk_hardware.OBS_LAYOUT)
    assert total == run_walk_hardware.OBS_DIM


def test_default_cli_frame_is_the_one_every_checkpoint_was_trained_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_walk_hardware.py"])
    assert run_walk_hardware.parse_args().policy_joint_frame == "canonical"
