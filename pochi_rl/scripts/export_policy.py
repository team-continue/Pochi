"""Export an RSL-RL actor checkpoint to TorchScript and ONNX."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from pochi_rl.policy_io import find_actor
from pochi_rl.task_spec import POCHI_TASK_SPEC


class ActorWrapper(torch.nn.Module):
  def __init__(self, actor: torch.nn.Module):
    super().__init__()
    self.actor = actor

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.actor(obs)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=Path("exports"))
  parser.add_argument(
    "--obs-dim", type=int, default=POCHI_TASK_SPEC.observations.policy_dim
  )
  parser.add_argument("--opset", type=int, default=17)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  args.output.mkdir(parents=True, exist_ok=True)

  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  actor = find_actor(checkpoint).eval()
  wrapped = ActorWrapper(actor).eval()
  example = torch.zeros(1, args.obs_dim, dtype=torch.float32)

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  stem = f"pochi_velocity_flat_{stamp}"
  torchscript_path = args.output / f"{stem}.pt"
  onnx_path = args.output / f"{stem}.onnx"

  scripted = torch.jit.trace(wrapped, example)
  scripted.save(torchscript_path)

  torch.onnx.export(
    wrapped,
    example,
    onnx_path,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=args.opset,
  )

  print(f"Wrote {torchscript_path}")
  print(f"Wrote {onnx_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
