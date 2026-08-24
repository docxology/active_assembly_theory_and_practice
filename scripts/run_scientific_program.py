#!/usr/bin/env python3
"""Unified offline-first orchestration for the scientific promotion tranche."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compiler_transfer import write_compiler_transfer  # noqa: E402
from src.external_sources import (  # noqa: E402
    EXTERNAL_MANIFEST_PATH,
    acquire_external_manifest,
    write_external_source_validation_payload,
)
from src.formal_correspondence import write_formal_correspondence  # noqa: E402
from src.intervention_analysis import write_intervention_power_payload  # noqa: E402
from src.quotient_classifier import write_quotient_classifier_report  # noqa: E402
from src.replication import (  # noqa: E402
    PUBLIC_MANIFEST,
    run_label_blind_model,
    score_replication_predictions,
    validate_replication_bundle,
    write_replication_bundle,
)
from src.scientific_protocol import write_scientific_protocol  # noqa: E402
from src.sensor_costs import write_sensor_cost_payload  # noqa: E402
from src.theorem_phase import write_theorem_or_countertheorem  # noqa: E402


def _path(value: str | None) -> Path:
    return PROJECT_ROOT if value is None else Path(value).resolve()


def _write_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("protocol", help="write protocol v2")
    subparsers.add_parser("formal", help="write bounded formal correspondence")
    subparsers.add_parser("benchmark", help="write quotient and intervention receipts")
    subparsers.add_parser("transfer", help="write compiler capability receipt")
    subparsers.add_parser("sensing", help="write measured sensing-cost receipt")
    subparsers.add_parser("theorem", help="write theorem/countertheorem receipt")
    subparsers.add_parser("external", help="write offline external-source receipt")
    acquire = subparsers.add_parser(
        "acquire-external",
        help="acquire only pinned sources with explicit network opt-in",
    )
    acquire.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / EXTERNAL_MANIFEST_PATH
    )
    acquire.add_argument(
        "--destination", type=Path, default=PROJECT_ROOT / "output/external_sources"
    )
    acquire.add_argument("--allow-network", action="store_true")

    subparsers.add_parser(
        "generate-public", help="generate label-blind public replication inputs"
    )
    subparsers.add_parser(
        "verify-public", help="verify public/private replication separation"
    )
    replay = subparsers.add_parser(
        "replay", help="offline deterministic replication replay"
    )
    replay.add_argument("--offline", action="store_true", default=True)
    score = subparsers.add_parser(
        "score", help="score predictions against sealed gold labels"
    )
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--gold-path", type=Path)
    subparsers.add_parser("verify-separation", help="alias for verify-public")
    run_model = subparsers.add_parser(
        "run-model", help="run a deterministic label-blind model on public inputs"
    )
    run_model.add_argument(
        "--model",
        choices=(
            "calibrated_interpretable",
            "temporal_probabilistic",
            "neural_sequence",
        ),
        default="calibrated_interpretable",
    )
    run_model.add_argument("--public-path", type=Path)
    run_model.add_argument("--output", type=Path)
    subparsers.add_parser("write-receipts", help="write the complete local receipt set")

    args = parser.parse_args()
    root = _path(args.project_root)
    command = args.command
    if command == "protocol":
        print(write_scientific_protocol(root))
    elif command == "formal":
        print(write_formal_correspondence(root))
    elif command == "benchmark":
        print(write_quotient_classifier_report(root))
        print(write_intervention_power_payload(root))
    elif command == "transfer":
        print(write_compiler_transfer(root))
    elif command == "sensing":
        print(write_sensor_cost_payload(root))
    elif command == "theorem":
        print(write_theorem_or_countertheorem(root))
    elif command == "external":
        print(write_external_source_validation_payload(root))
    elif command == "acquire-external":
        result = acquire_external_manifest(
            args.manifest, args.destination, allow_network=args.allow_network
        )
        _write_json(result)
        return 0 if result["status"] in {"validated", "blocked_external"} else 1
    elif command in {"generate-public", "replay"}:
        print(write_replication_bundle(root))
        valid, detail = validate_replication_bundle(root)
        print(detail)
        return 0 if valid else 1
    elif command in {"verify-public", "verify-separation"}:
        valid, detail = validate_replication_bundle(root)
        print(detail)
        return 0 if valid else 1
    elif command == "run-model":
        public_path = args.public_path or (root / PUBLIC_MANIFEST)
        output_path = args.output or (
            root / "output" / "replication" / f"{args.model}_predictions.json"
        )
        print(run_label_blind_model(public_path, output_path, model=args.model))
    elif command == "score":
        _write_json(
            score_replication_predictions(
                args.predictions, root, gold_path=args.gold_path
            )
        )
    elif command == "write-receipts":
        paths = [
            write_scientific_protocol(root),
            write_formal_correspondence(root),
            write_quotient_classifier_report(root),
            write_intervention_power_payload(root),
            write_compiler_transfer(root),
            write_sensor_cost_payload(root),
            write_theorem_or_countertheorem(root),
            write_external_source_validation_payload(root),
            write_replication_bundle(root),
        ]
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
