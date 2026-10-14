"""Check an exported policy on CPU without connecting to robot hardware."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    args = parser.parse_args()
    contract = json.loads((args.export_dir / "policy_contract.json").read_text())
    path = args.export_dir / contract["model"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != contract["model_sha256"]:
        raise RuntimeError("Policy checksum differs from its contract")
    torch.set_num_threads(1)
    policy = torch.jit.load(str(path), map_location="cpu").eval()
    results = []
    with torch.inference_mode():
        for batch in (1, 9):
            obs = torch.zeros(batch, 50, dtype=torch.float32)
            obs[:, 8] = -1.0
            if batch > 1:
                obs[1:, 9] = 0.25
                obs[1:, 49] = 1.0
            action = policy(obs)
            if tuple(action.shape) != (batch, 12):
                raise RuntimeError(f"Unexpected action shape: {action.shape}")
            if action.dtype != torch.float32 or not torch.isfinite(action).all():
                raise RuntimeError("Action must be finite float32")
            results.append({"batch": batch, "output_shape": list(action.shape)})
    print(json.dumps({"variant": contract.get("variant"), "sha256": digest,
                      "torch_version": torch.__version__, "cpu_inference": results,
                      "hardware_commands_sent": False}, indent=2))


if __name__ == "__main__":
    main()
