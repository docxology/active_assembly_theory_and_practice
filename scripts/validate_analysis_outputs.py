"""Regenerate and validate analysis outputs."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.claim_register import write_claim_register
from src.experiments import run_experiment
from src.figures import plot_inference_diagnostics
from src.redteam_report import write_redteam_report
from src.reproducibility import write_reproducibility_manifest
from src.rxinfer_bridge import run_rxinfer_backend
from src.scholarship import write_scholarship_sources
from src.validation import write_validation_report


def main() -> int:
    experiment = run_experiment()
    rxinfer_result = run_rxinfer_backend(
        PROJECT_ROOT / experiment.data_paths["trace_bundle"]
    )
    plot_inference_diagnostics(
        list(experiment.inference.free_energy_by_step),
        list(rxinfer_result.free_energy_values),
        PROJECT_ROOT / "output" / "figures" / "inference_diagnostics.png",
    )
    write_scholarship_sources()
    write_claim_register()
    write_redteam_report()
    write_reproducibility_manifest()
    report = write_validation_report()
    print("validation_passed:", report.summary["passed"])
    print("validation_failed:", report.summary["failed"])
    print("validation_total:", report.summary["total"])
    for path in report.artifact_paths.values():
        print("artifact:", PROJECT_ROOT / path)
    return 0 if report.summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
