"""Capture actor inputs from the unchanged historical C2c frozen trace worker.

Run from the historical checkout with the worker's arguments. No training entry
point is called; hooks only copy inputs and outputs of deterministic inference.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
from rsl_rl.runners import OnPolicyRunner


def main() -> None:
    root = Path.cwd()
    path = root / "scripts/flapping_rl/compare_pure_rl_longitudinal_checkpoints.py"
    spec = importlib.util.spec_from_file_location("historical_trace_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._parse_worker_args(sys.argv[1:])
    output = args.trace_output.with_suffix(".npz")
    if output.exists():
        raise FileExistsError(output)
    observations, actions = [], []
    original = OnPolicyRunner.get_inference_policy
    original_write = module._write_csv

    def write_capture(path, rows):
        original_write(path, rows)
        if not observations:
            raise RuntimeError("No inference captured")
        # Kit close can terminate the process; persist before closing the app.
        np.savez_compressed(output, policy_input=np.concatenate(observations),
                            raw_action=np.concatenate(actions))
        print(f"Captured {len(observations)} frozen policy inputs: {output}", flush=True)

    module._write_csv = write_capture

    def capture_policy(self, *positional, **keywords):
        policy = original(self, *positional, **keywords)
        network = self.alg.policy if hasattr(self.alg, "policy") else self.alg.actor_critic

        def hook(_module, inputs, result):
            observations.append(inputs[0].detach().cpu().numpy().copy())
            actions.append(result.detach().cpu().numpy().copy())

        network.actor.register_forward_hook(hook)
        return policy

    OnPolicyRunner.get_inference_policy = capture_policy
    try:
        module._run_worker(sys.argv[1:])
    finally:
        OnPolicyRunner.get_inference_policy = original


if __name__ == "__main__":
    main()
