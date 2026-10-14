import pytest

from pochi_rl.mjlab.viser_keyboard import HeldKeyController


def test_six_directions_combination_cancellation_and_release():
  control = HeldKeyController()
  for sequence, (keys, expected) in enumerate(
    [
      (["w"], (0.5, 0, 0)),
      (["s"], (-0.5, 0, 0)),
      (["a"], (0, 0.3, 0)),
      (["d"], (0, -0.3, 0)),
      (["q"], (0, 0, 0.6)),
      (["e"], (0, 0, -0.6)),
      (["w", "a", "q"], (0.5, 0.3, 0.6)),
      (["w", "s", "q", "e"], (0, 0, 0)),
      ([], (0, 0, 0)),
    ]
  ):
    assert control.receive("browser", keys, sequence)
    assert control.command() == expected


def test_idle_browser_cannot_cancel_other_browser_and_disconnect_stops():
  control = HeldKeyController()
  assert control.receive("first", ["w"], 0)
  assert control.receive("second", [], 0)
  assert control.command() == (0.5, 0, 0)
  assert not control.receive("second", ["s"], 1)
  control.disconnect("first")
  assert control.command() == (0, 0, 0)
  assert control.receive("second", ["s"], 2)
  assert control.command() == (-0.5, 0, 0)


def test_stale_connection_times_out_even_without_close_packet():
  now = [0.0]
  control = HeldKeyController(clock=lambda: now[0])
  assert control.receive("browser", ["w"], 0)
  now[0] = 0.49
  assert control.command() == (0.5, 0, 0)
  now[0] = 0.5
  assert control.command() == (0, 0, 0)


def test_zero_button_latches_until_release_and_ignores_old_packets():
  control = HeldKeyController()
  assert control.receive("browser", ["w"], 0)
  control.stop()
  assert not control.receive("browser", ["w"], 1)
  assert control.command() == (0, 0, 0)
  assert control.receive("browser", [], 2)
  assert not control.receive("browser", ["w"], 1)
  assert control.receive("browser", ["w"], 3)
  assert control.command() == (0.5, 0, 0)
  control.enabled = False
  assert control.command() == (0, 0, 0)


@pytest.mark.parametrize(
  "keys,sequence",
  [
    (["x"], 0),
    ("w", 0),
    (["w"], True),
    (["w"], -1),
    (["w"], 0.5),
  ],
)
def test_invalid_packets_do_not_change_commands(keys, sequence):
  control = HeldKeyController()
  with pytest.raises(ValueError):
    control.receive("browser", keys, sequence)
  assert control.command() == (0, 0, 0)
