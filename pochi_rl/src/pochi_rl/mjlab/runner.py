"""Pochi-specific RSL-RL runner customizations."""

from __future__ import annotations

import atexit
import os
import signal
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from mjlab.rl import MjlabOnPolicyRunner

_ATEXIT_SIGNAL_GUARD_INSTALLED = False


def _ignore_termination_signals() -> None:
  for sig_name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
    sig = getattr(signal, sig_name, None)
    if sig is None:
      continue
    try:
      signal.signal(sig, signal.SIG_IGN)
    except (OSError, ValueError):
      pass


def _install_atexit_signal_guard() -> None:
  """Stop torch-elastic SignalException from interrupting atexit cleanup.

  torchrunx sends SIGTERM to workers on shutdown, and torch elastic installs
  signal handlers that raise SignalException. If a SIGTERM lands while an
  atexit-registered cleanup hook (e.g. a TemporaryDirectory rmtree) is still
  running, the exception aborts it and is logged as a noisy "Exception ignored
  in atexit callback". Registering this guard early means it runs first via
  LIFO order and silences these signals before later atexit hooks execute.
  """
  global _ATEXIT_SIGNAL_GUARD_INSTALLED
  if _ATEXIT_SIGNAL_GUARD_INSTALLED:
    return
  atexit.register(_ignore_termination_signals)
  _ATEXIT_SIGNAL_GUARD_INSTALLED = True


def _skip_initial_parameter_broadcast(self) -> None:
  """Skip rsl-rl's initial object broadcast.

  The model seed is normalized across ranks before runner construction, so all
  ranks initialize identical parameters without needing the broadcast that hangs
  with this torchrunx/NCCL stack.
  """

  return None


def _reduce_parameters_gloo(self) -> None:
  all_params = list(self.actor.parameters()) + list(self.critic.parameters())
  if self.rnd:
    all_params += list(self.rnd.parameters())

  for param in all_params:
    if param.grad is None:
      continue
    grad = param.grad.detach().cpu()
    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self._pochi_gloo_group)
    grad /= self.gpu_world_size
    param.grad.copy_(grad.to(device=param.grad.device))


def _to_cpu(value: Any) -> Any:
  if torch.is_tensor(value):
    return value.detach().cpu()
  if isinstance(value, dict):
    return {key: _to_cpu(child) for key, child in value.items()}
  if isinstance(value, list):
    return [_to_cpu(child) for child in value]
  if isinstance(value, tuple):
    return tuple(_to_cpu(child) for child in value)
  return value


class PochiOnPolicyRunner(MjlabOnPolicyRunner):
  """On-policy runner with robust multi-GPU parameter synchronization."""

  def __init__(self, *args, **kwargs) -> None:
    if len(args) >= 2 and int(os.getenv("WORLD_SIZE", "1")) > 1:
      train_cfg = args[1]
      local_rank = int(os.getenv("LOCAL_RANK", "0"))
      torch.manual_seed(int(train_cfg["seed"]) - local_rank)
      train_cfg["algorithm"]["schedule"] = "fixed"

    _install_atexit_signal_guard()

    super().__init__(*args, **kwargs)
    self.alg.broadcast_parameters = MethodType(
      _skip_initial_parameter_broadcast,
      self.alg,
    )
    if self.is_distributed:
      self.alg._pochi_gloo_group = dist.new_group(backend="gloo")
      self.alg.reduce_parameters = MethodType(_reduce_parameters_gloo, self.alg)

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}

    if torch.cuda.is_available():
      torch.cuda.synchronize()

    saved_dict = _to_cpu(self.alg.save())
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = _to_cpu(infos)
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)
