"""Pochi RL task package."""

# Importing this package should populate mjlab's task registry when installed via
# the `mjlab.tasks` entry point.
try:
  from pochi_rl.mjlab import tasks as _tasks  # noqa: F401
except ModuleNotFoundError as exc:
  if exc.name != "mjlab":
    raise

__all__ = []
