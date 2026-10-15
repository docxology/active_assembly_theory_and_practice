from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .fixture_sensor_environment import FixtureSensorEnvironment
from .gnn_experiments import ROOT, envelope, run_gnn_study, run_passive, write_payload
from .gnn_model import GNNError
from .gnn_upstream import compile_upstream
from .gnn_validation import validate_gnn_study
from .pymdp_runtime import run_pymdp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute and audit upstream-parsed GNN models."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "study", "audit", "native"):
        command = commands.add_parser(name)
        command.add_argument(
            "--gnn-root", type=Path, default=os.environ.get("GNN_ROOT")
        )
        if name in ("validate", "run"):
            command.add_argument("--model", type=Path, required=True)
        if name != "validate":
            command.add_argument(
                "--output-dir",
                type=Path,
                default=ROOT
                / ("output/gnn/individual" if name == "run" else "output/gnn"),
            )
        if name == "run":
            command.add_argument(
                "--fixture", choices=("x86_accumulator", "x86_bitwise"), required=True
            )
            command.add_argument("--steps", type=int, default=1)
            command.add_argument("--seed", type=int)
        if name == "audit":
            command.add_argument("--require-native", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.gnn_root is None:
            raise GNNError("GNN_UPSTREAM_REQUIRED: set GNN_ROOT or --gnn-root")
        if args.command == "study":
            run_gnn_study(args.output_dir, args.gnn_root)
            errors = validate_gnn_study(args.output_dir, args.gnn_root)
            if errors:
                raise RuntimeError("; ".join(errors))
            print("models: passive_phase fixture_sensor")
        elif args.command == "audit":
            errors = validate_gnn_study(
                args.output_dir, args.gnn_root, require_native=args.require_native
            )
            print(
                json.dumps(
                    {
                        "status": "failed" if errors else "validated_bounded",
                        "errors": errors,
                    }
                )
            )
            return int(bool(errors))
        elif args.command == "native":
            from .rxinfer_bridge import run_shared_gnn_native

            errors = validate_gnn_study(args.output_dir, args.gnn_root)
            if errors:
                raise ValueError("; ".join(errors))
            model = compile_upstream(
                ROOT / "configs/gnn/passive_phase.md", args.gnn_root
            ).model
            trace = json.loads((args.output_dir / "passive_run.json").read_text())
            result = run_shared_gnn_native(
                model,
                [row["observation"][0] for row in trace["trajectory"]],
                args.output_dir / "native_backend",
            )
            write_payload(args.output_dir / "native_comparison.json", result)
            print(json.dumps({"max_abs_error": result["max_abs_error"]}))
        else:
            qualified = compile_upstream(args.model, args.gnn_root)
            model = qualified.model
            if args.command == "validate":
                print(json.dumps(qualified.receipt, sort_keys=True))
                return 0
            seed = model.seed if args.seed is None else args.seed
            if len(model.factors) == 1:
                result = run_passive(model, args.fixture, steps=args.steps, seed=seed)
            else:
                result = run_pymdp(
                    model,
                    FixtureSensorEnvironment(args.fixture),
                    steps=args.steps,
                    seed=seed,
                )
            write_payload(args.output_dir / "compiled_model.json", model.to_dict())
            write_payload(args.output_dir / "upstream.json", qualified.receipt)
            write_payload(args.output_dir / "run.json", envelope(result))
        return 0
    except (GNNError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1
