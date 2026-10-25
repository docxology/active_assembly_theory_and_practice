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
    EXTERNAL_MANIFEST_PATH,
    prepare_external_summary,
    write_compiler_capabilities,
    write_external_static_validation,
    write_external_corpus_preparation_audit,
    write_external_acquisition_payload,
    write_external_source_validation_payload,
)
from src.external_labeling import (  # noqa: E402
    write_external_label_contract,
)
from src.counterexample_search import write_counterexample_search  # noqa: E402
from src.formal_correspondence import write_formal_correspondence  # noqa: E402
from src.intervention_analysis import write_intervention_power_payload  # noqa: E402
from src.quotient_classifier import write_quotient_classifier_report  # noqa: E402
from src.replay_vm import (  # noqa: E402
    attest_and_ingest_vm_replay,
    create_replay_vm,
    provision_replay_vm,
    replay_in_vm,
    stop_replay_vm,
    vm_status,
    vm_paths,
    write_vm_manifest,
)
from src.receipt import promote_evidence_summary  # noqa: E402
from src.replication import (  # noqa: E402
    GOLD_LABELS,
    PUBLIC_MANIFEST,
    _public_bundle_digest,
    ingest_independent_replay,
    run_label_blind_model,
    score_replication_predictions,
    validate_public_replay_bundle,
    validate_replication_bundle,
    write_replay_receipt,
    write_replication_bundle,
)
from src.scientific_protocol import write_scientific_protocol  # noqa: E402
from src.sensor_costs import write_sensor_cost_payload  # noqa: E402
from src.theorem_phase import write_theorem_or_countertheorem  # noqa: E402
from src.sensing_stress import write_sensing_stress  # noqa: E402


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
    subparsers.add_parser(
        "counterexamples",
        help="write the formal counterexample-search receipt",
    )
    subparsers.add_parser(
        "sensing-stress",
        help="write the active-sensing stress-study receipt",
    )
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
        "audit-external-preparation",
        help=(
            "write the offline, rehearsal-only external preparation audit "
            "receipt (no network, no execution)"
        ),
    )
    subparsers.add_parser(
        "validate-external",
        help="compile acquired sources statically and write receipts",
    )
    subparsers.add_parser(
        "promote-evidence", help="summarize current evidence statuses without mutation"
    )
    generate_figures = subparsers.add_parser(
        "generate-figures", help="generate the source-bound figure bundle and atlas"
    )
    generate_figures.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf", "atlas"),
        default=("png", "svg", "pdf", "atlas"),
    )
    subparsers.add_parser(
        "audit-figures",
        help="validate figure specs, hashes, vectors, and offline atlas",
    )
    subparsers.add_parser(
        "render-manuscript",
        help="legacy alias for manuscript-variable generation and hydration only",
    )
    subparsers.add_parser(
        "vm-probe", help="write the blue-backed Lima VM capability/config receipt"
    )
    vm_create = subparsers.add_parser(
        "vm-create", help="create or start the pinned blue-backed replay VM"
    )
    vm_create.add_argument(
        "--allow-network",
        action="store_true",
        help="allow Lima to acquire the pinned guest image during first creation",
    )
    vm_provision = subparsers.add_parser(
        "vm-provision", help="provision the pinned guest environment"
    )
    vm_provision.add_argument(
        "--allow-network",
        action="store_true",
        help="allow pinned image/package acquisition during provisioning",
    )
    vm_replay = subparsers.add_parser(
        "vm-replay", help="run the public-only replay in the isolated VM namespace"
    )
    vm_replay.add_argument(
        "--model",
        choices=(
            "calibrated_interpretable",
            "temporal_probabilistic",
            "neural_sequence",
        ),
        default="calibrated_interpretable",
    )
    vm_replay.add_argument("--seed", type=int, default=0)
    subparsers.add_parser(
        "vm-attest-replay",
        help="attest the same-host VM boundary and protected-score its receipt",
    )
    subparsers.add_parser("vm-stop", help="stop the VM without deleting its disk")
    subparsers.add_parser("vm-status", help="show the current blue-backed VM status")
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
        "verify-public", help="verify the public replay bundle without gold labels"
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
    subparsers.add_parser(
        "verify-separation",
        help=(
            "verify the public replay bundle, upgrading to the stronger "
            "public/private separation check when gold labels are present"
        ),
    )
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
    receipt.add_argument(
        "--source-revision",
        help="optional committed source revision for a reproducible runtime archive",
    )
    receipt.add_argument(
        "--source-archive-sha256",
        help="optional digest of the runtime source archive bound to source revision",
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
    elif command == "counterexamples":
        print(write_counterexample_search(root))
    elif command == "sensing-stress":
        print(write_sensing_stress(root))
    elif command == "external":
        print(write_external_label_contract(root))
        print(write_external_source_validation_payload(root))
    elif command == "label-external":
        print(write_external_label_contract(root))
    elif command == "probe-capabilities":
        print(write_compiler_capabilities(root))
    elif command == "prepare-external":
        summary, ok = prepare_external_summary(root)
        _write_json(summary)
        return 0 if ok else 1
    elif command == "validate-external":
        capability_path = write_compiler_capabilities(root)
        label_path = write_external_label_contract(root)
        static_path = write_external_static_validation(root)
        external_path = write_external_source_validation_payload(root)
        print(capability_path)
        print(label_path)
        print(static_path)
        print(external_path)
    elif command == "audit-external-preparation":
        # Offline by construction: the writer accepts no network flag and
        # never touches a URL; it only rehearses the on-disk cache.
        print(write_external_corpus_preparation_audit(root))
        return 0
    elif command == "promote-evidence":
        _write_json(promote_evidence_summary(root))
        return 0
    elif command == "generate-figures":
        from src.advanced_figures import write_advanced_figures

        from src.figure_contract import validate_figure_bundle, write_figure_bundle

        # ``write_advanced_figures`` remains the full producer path; the
        # explicit bundle call honors the public format interface and refreshes
        # the canonical registries after all base figures exist.
        write_advanced_figures(root / "output")
        bundle = write_figure_bundle(root / "output", formats=args.formats)
        errors = validate_figure_bundle(root / "output")
        _write_json(
            {
                "status": "validated" if not errors else "failed",
                "formats": list(args.formats),
                "bundle": bundle,
                "errors": errors,
            }
        )
        return 0 if not errors else 1
    elif command == "audit-figures":
        from src.figure_contract import validate_figure_bundle

        errors = validate_figure_bundle(root / "output")
        _write_json(
            {"status": "validated" if not errors else "failed", "errors": errors}
        )
        return 0 if not errors else 1
    elif command == "render-manuscript":
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "z_generate_manuscript_variables.py"),
            ],
            cwd=root,
            check=False,
        )
        return int(result.returncode)
    elif command == "vm-probe":
        print(write_vm_manifest(root))
    elif command == "vm-create":
        print(create_replay_vm(root, allow_network=args.allow_network))
    elif command == "vm-provision":
        print(provision_replay_vm(root, allow_network=args.allow_network))
    elif command == "vm-replay":
        print(replay_in_vm(root, model=args.model, seed=args.seed))
    elif command == "vm-attest-replay":
        print(attest_and_ingest_vm_replay(root))
    elif command == "vm-stop":
        print(stop_replay_vm(root))
    elif command == "vm-status":
        _write_json(
            {
                "vm_root": str(vm_paths()["root"]),
                "status": vm_status(vm_paths()),
            }
        )
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
                "status": (
                    "failed_artifact_audit"
                    if artifact_errors
                    else "generated_local_pipeline"
                ),
                "variable_count": len(variables),
                "artifact_errors": artifact_errors,
            }
        )
        return 0 if not artifact_errors else 1
    elif command == "acquire-external":
        result = write_external_acquisition_payload(
            root,
            manifest_path=args.manifest,
            destination=args.destination,
            allow_network=args.allow_network,
        )
        print(result)
        payload = json.loads(result.read_text(encoding="utf-8"))
        return 0 if payload.get("status") in {"validated", "blocked_external"} else 1
    elif command == "generate-public":
        print(write_replication_bundle(root))
        valid, detail = validate_replication_bundle(root)
        print(detail)
        return 0 if valid else 1
    elif command == "replay":
        if not args.offline:
            print("replay requires explicit --offline", file=sys.stderr)
            return 2
        valid, detail = validate_public_replay_bundle(root)
        print(detail)
        return 0 if valid else 1
    elif command == "verify-public":
        valid, detail = validate_public_replay_bundle(root)
        print(detail)
        return 0 if valid else 1
    elif command == "verify-separation":
        if (root / GOLD_LABELS).is_file():
            valid, detail = validate_replication_bundle(root)
        else:
            valid, detail = validate_public_replay_bundle(root)
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
                source_revision=args.source_revision,
                source_archive_sha256=args.source_archive_sha256,
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
            write_counterexample_search(root),
            write_sensing_stress(root),
            write_compiler_capabilities(root),
            write_external_label_contract(root),
            write_external_static_validation(root),
            write_external_source_validation_payload(root),
            write_external_corpus_preparation_audit(root),
            write_replication_bundle(root),
        ]
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
