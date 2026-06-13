"""Run the native Julia/RxInfer backend on the generated trace bundle."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_experiment
from src.figures import plot_inference_diagnostics
from src.rxinfer_bridge import run_rxinfer_backend


def main() -> int:
    experiment = run_experiment()
    result = run_rxinfer_backend(PROJECT_ROOT / experiment.data_paths["trace_bundle"])
    diagnostic_figure = plot_inference_diagnostics(
        list(experiment.inference.free_energy_by_step),
        list(result.free_energy_values),
        PROJECT_ROOT / "output" / "figures" / "inference_diagnostics.png",
    )
    print("sample:", result.sample_id)
    print("model:", result.model_name)
    print("rxinfer_free_energy_values:", len(result.free_energy_values))
    print("rxinfer_final_free_energy:", result.final_free_energy)
    for path in [*result.artifact_paths.values(), *result.figure_paths.values()]:
        print("artifact:", PROJECT_ROOT / path)
    print("artifact:", diagnostic_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
