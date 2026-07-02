"""Python bridge for the native Julia/RxInfer inference backend.

The RxInfer/Julia numbers that flow into the manuscript (``JULIA_VERSION``,
``RXINFER_VERSION``, ``RXINFER_FINAL_FREE_ENERGY``) are only reproducible against
a known external toolchain. Just as ``pyproject.toml`` pins the Python
dependencies, the expected backend versions are pinned explicitly here as
:data:`EXPECTED_JULIA_VERSION` / :data:`EXPECTED_RXINFER_VERSION` and enforced by
:func:`validate_backend_versions`, so a silent toolchain bump fails loudly rather
than republishing numbers that no longer match the pinned environment.

The availability of that toolchain is also made explicit rather than implicit:
:func:`julia_available` is the documented contract callers use to choose between
a clean skip (Julia absent) and running the backend, while
:func:`resolve_julia_executable` raises a clearly-labeled ``RuntimeError`` when no
Julia executable can be found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from .figures import plot_rxinfer_free_energy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
JULIA_PROJECT = PROJECT_ROOT / "julia" / "ActiveAssemblyRxInfer"
JULIA_RUNNER = JULIA_PROJECT / "scripts" / "run_rxinfer.jl"

# Pinned native-backend toolchain versions. These mirror how ``pyproject.toml``
# pins the Python dependencies: the reproducible manuscript numbers depend on
# this exact toolchain, so the expected versions are declared explicitly and
# enforced against the backend diagnostics by ``validate_backend_versions``.
EXPECTED_JULIA_VERSION = "1.12.1"
EXPECTED_RXINFER_VERSION = "5.4.0"


@dataclass(frozen=True)
class RxInferRunResult:
    """Validated artifact summary returned by the native RxInfer backend."""

    run_id: str
    sample_id: str
    model_name: str
    mode: str
    free_energy_values: tuple[float, ...]
    final_free_energy: float
    data_hash: str
    config_hash: str
    rxinfer_version: str
    julia_version: str
    artifact_paths: dict[str, str]
    figure_paths: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe bridge result."""

        data = asdict(self)
        data["free_energy_values"] = list(self.free_energy_values)
        return data


def file_sha256(path: Path) -> str:
    """Return a SHA-256 hash for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_julia_executable(julia_executable: str = "julia") -> str:
    """Resolve a stable Julia executable for Python and template invocations."""

    requested = Path(julia_executable).expanduser()
    if requested.is_absolute() and requested.exists():
        return str(requested)
    if julia_executable == "julia":
        env_julia = os.environ.get("JULIA")
        if env_julia and Path(env_julia).expanduser().exists():
            return str(Path(env_julia).expanduser())
        juliaup = Path.home() / ".juliaup" / "bin" / "julia"
        if juliaup.exists():
            return str(juliaup)
    path_julia = shutil.which(julia_executable)
    if path_julia is None:
        raise RuntimeError(f"Julia executable not found: {julia_executable}")
    return path_julia


def julia_available(julia_executable: str = "julia") -> bool:
    """Return whether a usable Julia executable can be resolved.

    This is the explicit availability contract for the native RxInfer backend.
    Callers that want a *clean skip* when Julia is absent (rather than the hard
    failure :func:`run_rxinfer_backend` raises) gate on this before invoking the
    backend. It never raises: a missing toolchain returns ``False``.
    """

    try:
        resolve_julia_executable(julia_executable)
    except RuntimeError:
        return False
    return True


def validate_backend_versions(
    julia_version: str,
    rxinfer_version: str,
    *,
    expected_julia_version: str = EXPECTED_JULIA_VERSION,
    expected_rxinfer_version: str = EXPECTED_RXINFER_VERSION,
) -> None:
    """Fail loudly when the native backend drifts from the pinned toolchain.

    The manuscript reports ``JULIA_VERSION`` and ``RXINFER_VERSION`` verbatim from
    the backend diagnostics. A silent toolchain bump would republish numbers that
    no longer match the pinned, reproducible environment, so a mismatch is a
    hard, clearly-labeled ``RuntimeError`` rather than an implicit acceptance.

    Args:
        julia_version: Julia version reported by the backend diagnostics.
        rxinfer_version: RxInfer version reported by the backend diagnostics.
        expected_julia_version: Pinned Julia version (defaults to the module pin).
        expected_rxinfer_version: Pinned RxInfer version (defaults to the module pin).
    """

    mismatches: list[str] = []
    if julia_version != expected_julia_version:
        mismatches.append(
            f"Julia {julia_version!r} != pinned {expected_julia_version!r}"
        )
    if rxinfer_version != expected_rxinfer_version:
        mismatches.append(
            f"RxInfer {rxinfer_version!r} != pinned {expected_rxinfer_version!r}"
        )
    if mismatches:
        raise RuntimeError(
            "native RxInfer backend version mismatch: " + "; ".join(mismatches)
        )


def load_free_energy_records(path: Path) -> list[dict[str, object]]:
    """Load the RxInfer free-energy JSONL file."""

    if not path.exists():
        raise FileNotFoundError(f"missing RxInfer free-energy trace: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("RxInfer free-energy trace is empty")
    return records


def validate_free_energy_records(
    records: list[dict[str, object]],
    *,
    expected_data_hash: str | None = None,
) -> None:
    """Validate required free-energy fields, finiteness, and hash consistency."""

    required_fields = {
        "run_id",
        "sample_id",
        "model_name",
        "mode",
        "iteration",
        "observation_index",
        "value",
        "source_field",
        "is_finite",
        "delta",
        "rxinfer_version",
        "julia_version",
        "data_hash",
        "config_hash",
    }
    previous_iteration = 0
    for record in records:
        missing = sorted(required_fields - set(record))
        if missing:
            raise ValueError(f"RxInfer free-energy record missing fields: {missing}")
        value = float(record["value"])
        if not math.isfinite(value) or record["is_finite"] is not True:
            raise ValueError(f"non-finite RxInfer free-energy value: {value}")
        iteration = int(record["iteration"])
        if iteration <= previous_iteration:
            raise ValueError("RxInfer free-energy iterations must be strictly increasing")
        previous_iteration = iteration
        if record["source_field"] != "result.free_energy":
            raise ValueError(f"unexpected RxInfer free-energy source: {record['source_field']}")
        if expected_data_hash and record["data_hash"] != expected_data_hash:
            raise ValueError("RxInfer data_hash does not match trace bundle hash")


def validate_rxinfer_outputs(
    output_dir: Path,
    *,
    expected_data_hash: str | None = None,
) -> list[dict[str, object]]:
    """Validate the complete RxInfer artifact set and return free-energy records."""

    free_energy_path = output_dir / "free_energy_trace.jsonl"
    for filename in ("posteriors.json", "diagnostics.json", "model_config.json"):
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing RxInfer artifact: {path}")
        json.loads(path.read_text(encoding="utf-8"))
    records = load_free_energy_records(free_energy_path)
    validate_free_energy_records(records, expected_data_hash=expected_data_hash)
    return records


def run_rxinfer_backend(
    trace_bundle_path: Path,
    *,
    output_dir: Path | None = None,
    iterations: int = 8,
    julia_executable: str = "julia",
    instantiate: bool = True,
) -> RxInferRunResult:
    """Run native Julia/RxInfer inference and validate all returned artifacts."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    trace_bundle_path = trace_bundle_path.resolve()
    if not trace_bundle_path.exists():
        raise FileNotFoundError(f"missing trace bundle: {trace_bundle_path}")
    julia_path = resolve_julia_executable(julia_executable)
    rxinfer_dir = (output_dir or OUTPUT_DIR / "rxinfer").resolve()
    rxinfer_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = rxinfer_dir / "stdout.log"
    stderr_path = rxinfer_dir / "stderr.log"

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    if instantiate:
        instantiate_command = [
            julia_path,
            f"--project={JULIA_PROJECT}",
            "-e",
            "using Pkg; Pkg.instantiate()",
        ]
        instantiated = subprocess.run(
            instantiate_command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        stdout_chunks.append("## Pkg.instantiate()\n" + instantiated.stdout)
        stderr_chunks.append("## Pkg.instantiate()\n" + instantiated.stderr)
        if instantiated.returncode != 0:
            stdout_path.write_text("\n".join(stdout_chunks), encoding="utf-8")
            stderr_path.write_text("\n".join(stderr_chunks), encoding="utf-8")
            raise RuntimeError(
                "Julia/RxInfer environment instantiation failed with exit code "
                f"{instantiated.returncode}; see {stderr_path}"
            )

    command = [
        julia_path,
        f"--project={JULIA_PROJECT}",
        str(JULIA_RUNNER),
        "--trace",
        str(trace_bundle_path),
        "--output",
        str(rxinfer_dir),
        "--iterations",
        str(iterations),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_chunks.append("## run_rxinfer.jl\n" + completed.stdout)
    stderr_chunks.append("## run_rxinfer.jl\n" + completed.stderr)
    stdout_path.write_text("\n".join(stdout_chunks), encoding="utf-8")
    stderr_path.write_text("\n".join(stderr_chunks), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "Julia/RxInfer backend failed with exit code "
            f"{completed.returncode}; see {stderr_path}"
        )

    expected_data_hash = file_sha256(trace_bundle_path)
    records = validate_rxinfer_outputs(rxinfer_dir, expected_data_hash=expected_data_hash)
    diagnostics = json.loads((rxinfer_dir / "diagnostics.json").read_text(encoding="utf-8"))
    validate_backend_versions(
        str(diagnostics["julia_version"]),
        str(diagnostics["rxinfer_version"]),
    )
    free_energy_values = tuple(float(record["value"]) for record in records)
    figure_path = plot_rxinfer_free_energy(
        records,
        OUTPUT_DIR / "figures" / "rxinfer_free_energy.png",
    )
    artifact_paths = {
        "free_energy_trace": str((rxinfer_dir / "free_energy_trace.jsonl").relative_to(PROJECT_ROOT)),
        "posteriors": str((rxinfer_dir / "posteriors.json").relative_to(PROJECT_ROOT)),
        "diagnostics": str((rxinfer_dir / "diagnostics.json").relative_to(PROJECT_ROOT)),
        "model_config": str((rxinfer_dir / "model_config.json").relative_to(PROJECT_ROOT)),
        "stdout": str(stdout_path.relative_to(PROJECT_ROOT)),
        "stderr": str(stderr_path.relative_to(PROJECT_ROOT)),
    }
    return RxInferRunResult(
        run_id=str(diagnostics["run_id"]),
        sample_id=str(diagnostics["sample_id"]),
        model_name=str(diagnostics["model_name"]),
        mode=str(diagnostics["mode"]),
        free_energy_values=free_energy_values,
        final_free_energy=float(diagnostics["free_energy_final"]),
        data_hash=str(diagnostics["data_hash"]),
        config_hash=str(diagnostics["config_hash"]),
        rxinfer_version=str(diagnostics["rxinfer_version"]),
        julia_version=str(diagnostics["julia_version"]),
        artifact_paths=artifact_paths,
        figure_paths={"rxinfer_free_energy": str(figure_path.relative_to(PROJECT_ROOT))},
    )
