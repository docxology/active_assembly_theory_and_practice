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
from src.artifact_contract import (  # noqa: E402
    seal_data_artifact_surface,
    validate_data_artifact_surface,
)
from src.external_sources import (  # noqa: E402
    EXTERNAL_CAPABILITY_ARTIFACT,
    EXTERNAL_MANIFEST_PATH,
    EXTERNAL_STATIC_RECEIPT_ARTIFACT,
    acquire_external_manifest,
    load_external_manifest,
    validate_external_manifest,
    write_compiler_capabilities,
    write_external_static_validation,
    write_external_source_validation_payload,
)
from src.external_corpus import (  # noqa: E402
    LABEL_CONTRACT_PATH,
    label_contract_summary,
    load_label_contracts,
    validate_label_contracts,
)
from src.external_labeling import (  # noqa: E402
    LABEL_CONTRACT_ARTIFACT,
    write_external_label_contract,
)
from src.formal_correspondence import write_formal_correspondence  # noqa: E402
from src.intervention_analysis import write_intervention_power_payload  # noqa: E402
from src.quotient_classifier import write_quotient_classifier_report  # noqa: E402
from src.replication import (  # noqa: E402
    INDEPENDENT_REPLAY_ARTIFACT,
    PUBLIC_MANIFEST,
    _public_bundle_digest,
    ingest_independent_replay,
    run_label_blind_model,
    score_replication_predictions,
    validate_replication_bundle,
    write_replay_receipt,
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
    subparsers.add_parser(
        "label-external",
        help="generate source-bound external pair labels and static receipts",
    )
    subparsers.add_parser(
        "probe-capabilities", help="probe locally compilable static targets"
    )
    subparsers.add_parser(
        "prepare-external", help="validate pinned manifest and label-contract inputs"
    )
    subparsers.add_parser(
        "validate-external",
        help="compile acquired sources statically and write receipts",
    )
    subparsers.add_parser(
        "promote-evidence", help="summarize current evidence statuses without mutation"
    )
    subparsers.add_parser(
        "seal-artifacts", help="migrate legacy project data JSON into receipt envelopes"
    )
    subparsers.add_parser(
        "audit-artifacts", help="fail-closed audit of the project data JSON surface"
    )
    subparsers.add_parser(
        "all", help="run the canonical source-bound experiment and manuscript producers"
    )
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
    replay.add_argument("--offline", action="store_true")
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
    receipt = subparsers.add_parser(
        "write-replay-receipt",
        help="write a pending separately operated replay receipt",
    )
    receipt.add_argument("--predictions", type=Path, required=True)
    receipt.add_argument("--output", type=Path, required=True)
    receipt.add_argument("--operator-id", required=True)
    receipt.add_argument("--machine-id", required=True)
    receipt.add_argument("--bundle-sha256", required=True)
    receipt.add_argument("--model", required=True)
    receipt.add_argument("--seed", type=int, required=True)
    receipt.add_argument("--public-path", type=Path)
    receipt.add_argument(
        "--started-at",
        help="recorded ISO-8601 start timestamp with timezone; defaults to now",
    )
    receipt.add_argument(
        "--completed-at",
        help="recorded ISO-8601 completion timestamp with timezone; defaults to now",
    )
    # REMAINDER preserves command flags such as ``replay --offline`` in the
    # receipt rather than asking the outer parser to understand the operator's
    # replay command.  The receipt producer still validates every token.
    receipt.add_argument(
        "--command", dest="replay_command", nargs=argparse.REMAINDER, required=True
    )
    ingest = subparsers.add_parser(
        "ingest-replay", help="ingest and protected-score a separate replay receipt"
    )
    ingest.add_argument("--receipt", type=Path, required=True)
    ingest.add_argument("--predictions", type=Path, required=True)
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
        print(write_external_label_contract(root))
        print(write_external_source_validation_payload(root))
    elif command == "label-external":
        print(write_external_label_contract(root))
    elif command == "probe-capabilities":
        print(write_compiler_capabilities(root))
    elif command == "prepare-external":
        manifest = load_external_manifest(root / EXTERNAL_MANIFEST_PATH)
        contracts = load_label_contracts(root / LABEL_CONTRACT_PATH)
        manifest_valid, manifest_detail = validate_external_manifest(manifest)
        contracts_valid, contracts_detail = validate_label_contracts(contracts)
        _write_json(
            {
                "status": "validated"
                if manifest_valid and contracts_valid
                else "failed",
                "manifest": manifest_detail,
                "label_contracts": contracts_detail,
                "label_summary": label_contract_summary(root),
                "network_used": False,
                "execution_performed": False,
            }
        )
        return 0 if manifest_valid and contracts_valid else 1
    elif command == "validate-external":
        capability_path = write_compiler_capabilities(root)
        label_path = write_external_label_contract(root)
        static_path = write_external_static_validation(root)
        external_path = write_external_source_validation_payload(root)
        print(capability_path)
        print(label_path)
        print(static_path)
        print(external_path)
        return 0
    elif command == "promote-evidence":
        status_paths = {
            "P13_external_validation": root
            / "output/data/external_source_validation.json",
            "P13_source_bound_labels": root / LABEL_CONTRACT_ARTIFACT,
            "P13_static_only": root / EXTERNAL_STATIC_RECEIPT_ARTIFACT,
            "P13_capabilities": root / EXTERNAL_CAPABILITY_ARTIFACT,
            "P15_local_replay_harness": root / "output/data/replication_manifest.json",
            "P15_independent_replay": root / INDEPENDENT_REPLAY_ARTIFACT,
        }
        rows: dict[str, object] = {}
        for name, path in status_paths.items():
            if not path.is_file():
                rows[name] = {"status": "not_yet_observed", "path": str(path)}
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                rows[name] = {"status": "failed", "path": str(path), "detail": str(exc)}
                continue
            rows[name] = {
                "status": payload.get("status", "not_yet_observed"),
                "endpoint_status": payload.get("endpoint_status"),
                "path": str(path),
            }
        _write_json({"status": "read_only_summary", "surfaces": rows})
        return 0
    elif command == "seal-artifacts":
        _write_json({"changed": seal_data_artifact_surface(root)})
        return 0 if not validate_data_artifact_surface(root) else 1
    elif command == "audit-artifacts":
        errors = validate_data_artifact_surface(root)
        _write_json(
            {"status": "validated" if not errors else "failed", "errors": errors}
        )
        return 0 if not errors else 1
    elif command == "all":
        from src.manuscript_variables import generate_variables

        variables = generate_variables()
        artifact_errors = validate_data_artifact_surface(root)
        _write_json(
            {
                "status": "validated_local_pipeline",
                "variable_count": len(variables),
                "artifact_errors": artifact_errors,
            }
        )
        return 0 if not artifact_errors else 1
    elif command == "acquire-external":
        result = acquire_external_manifest(
            args.manifest, args.destination, allow_network=args.allow_network
        )
        _write_json(result)
        return 0 if result["status"] in {"validated", "blocked_external"} else 1
    elif command == "generate-public":
        print(write_replication_bundle(root))
        valid, detail = validate_replication_bundle(root)
        print(detail)
        return 0 if valid else 1
    elif command == "replay":
        if not args.offline:
            print("replay requires explicit --offline", file=sys.stderr)
            return 2
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
    elif command == "write-replay-receipt":
        public_path = args.public_path or (root / PUBLIC_MANIFEST)
        bundle_hash = args.bundle_sha256
        if bundle_hash == "current":
            bundle_hash = _public_bundle_digest(root)
        print(
            write_replay_receipt(
                public_path,
                args.predictions,
                args.output,
                operator_id=args.operator_id,
                machine_id=args.machine_id,
                replay_bundle_sha256=bundle_hash,
                model=args.model,
                seed=args.seed,
                command=args.replay_command,
                project_root=root,
                started_at=args.started_at,
                completed_at=args.completed_at,
            )
        )
    elif command == "ingest-replay":
        print(ingest_independent_replay(args.receipt, args.predictions, root))
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
            write_compiler_capabilities(root),
            write_external_label_contract(root),
            write_external_static_validation(root),
            write_external_source_validation_payload(root),
            write_replication_bundle(root),
        ]
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
