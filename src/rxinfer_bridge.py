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

The committed Julia ``Project.toml``/``Manifest.toml`` pair is part of that
contract. :func:`validate_julia_manifest` checks the manifest's Julia and
RxInfer entries before either native producer is launched, so a stale lockfile
cannot be mistaken for a reproducible backend merely because a compatible
Julia executable happens to be installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tomllib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .figures import plot_rxinfer_categorical_free_energy, plot_rxinfer_free_energy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
JULIA_PROJECT = PROJECT_ROOT / "julia" / "ActiveAssemblyRxInfer"
JULIA_MANIFEST = JULIA_PROJECT / "Manifest.toml"
JULIA_RUNNER = JULIA_PROJECT / "scripts" / "run_rxinfer.jl"

# Pinned native-backend toolchain versions. These mirror how ``pyproject.toml``
# pins the Python dependencies: the reproducible manuscript numbers depend on
# this exact toolchain, so the expected versions are declared explicitly and
# enforced against the backend diagnostics by ``validate_backend_versions``.
EXPECTED_JULIA_VERSION = "1.12.7"
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


def _pinned_juliaup_binary(expected_version: str) -> str | None:
    """Return the exact-pinned Julia binary if a juliaup install matches.

    juliaup install layouts place each version at
    ``~/.julia/juliaup/julia-<version>+<triple>/bin/julia``. When the pinned
    backend version is present there, we prefer it over the generic ``julia``
    launcher, whose ``release`` channel can auto-advance (e.g. 1.12.1 -> 1.12.7)
    and otherwise trip the deliberate version-pin guard in
    :func:`validate_backend_versions`. The guard still fails loudly when the
    pinned version is genuinely absent.
    """

    base = Path.home() / ".julia" / "juliaup"
    if not base.is_dir():
        return None
    for candidate in sorted(base.glob(f"julia-{expected_version}*")):
        binary = candidate / "bin" / "julia"
        if binary.is_file():
            return str(binary)
    return None


def resolve_julia_executable(julia_executable: str = "julia") -> str:
    """Resolve a stable Julia executable for Python and template invocations.

    When ``julia_executable == "julia"`` the resolution order honours an explicit
    ``JULIA`` environment override, then the exact-pinned juliaup install (see
    :func:`_pinned_juliaup_binary`), then the generic ``~/.juliaup/bin/julia``
    launcher (whose ``release`` channel may auto-advance), and finally the
    ``PATH`` lookup. An explicit absolute path or a named channel is used
    directly. The returned binary is the one :func:`validate_backend_versions`
    checks against :data:`EXPECTED_JULIA_VERSION`.
    """

    requested = Path(julia_executable).expanduser()
    if requested.is_absolute() and requested.exists():
        return str(requested)
    if julia_executable == "julia":
        env_julia = os.environ.get("JULIA")
        if env_julia and Path(env_julia).expanduser().exists():
            return str(Path(env_julia).expanduser())
        pinned = _pinned_juliaup_binary(EXPECTED_JULIA_VERSION)
        if pinned is not None:
            return pinned
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


def validate_julia_manifest(
    manifest_path: Path = JULIA_MANIFEST,
    *,
    expected_julia_version: str = EXPECTED_JULIA_VERSION,
    expected_rxinfer_version: str = EXPECTED_RXINFER_VERSION,
) -> None:
    """Validate the committed Julia lockfile against the backend version pins.

    This check is intentionally structural and local: it does not resolve a
    package environment or infer a package's version from an executable. It
    verifies that the exact manifest selected by ``--project`` declares the
    same Julia and RxInfer versions that the bridge later requires the native
    diagnostics to report. A malformed, missing, or ambiguous manifest fails
    closed with a labeled ``RuntimeError``.

    Args:
        manifest_path: Manifest to inspect; defaults to this project's committed
            native-backend manifest.
        expected_julia_version: Julia version pinned by the bridge.
        expected_rxinfer_version: RxInfer version pinned by the bridge.
    """

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"native Julia manifest not found: {manifest_path}")
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(
            f"native Julia manifest is unreadable or malformed: {manifest_path}"
        ) from exc

    declared_julia = manifest.get("julia_version")
    if declared_julia != expected_julia_version:
        raise RuntimeError(
            "native Julia manifest version mismatch: "
            f"Julia {declared_julia!r} != pinned {expected_julia_version!r}"
        )

    dependencies = manifest.get("deps")
    rxinfer_entries = (
        dependencies.get("RxInfer") if isinstance(dependencies, dict) else None
    )
    if not isinstance(rxinfer_entries, list) or len(rxinfer_entries) != 1:
        raise RuntimeError(
            "native Julia manifest must contain exactly one [[deps.RxInfer]] entry"
        )
    declared_rxinfer = rxinfer_entries[0].get("version")
    if declared_rxinfer != expected_rxinfer_version:
        raise RuntimeError(
            "native Julia manifest RxInfer version mismatch: "
            f"RxInfer {declared_rxinfer!r} != pinned {expected_rxinfer_version!r}"
        )


def load_free_energy_records(path: Path) -> list[dict[str, Any]]:
    """Load the RxInfer free-energy JSONL file."""

    if not path.exists():
        raise FileNotFoundError(f"missing RxInfer free-energy trace: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not records:
        raise ValueError("RxInfer free-energy trace is empty")
    return records


def validate_free_energy_records(
    records: list[dict[str, Any]],
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
            raise ValueError(
                "RxInfer free-energy iterations must be strictly increasing"
            )
        previous_iteration = iteration
        if record["source_field"] != "result.free_energy":
            raise ValueError(
                f"unexpected RxInfer free-energy source: {record['source_field']}"
            )
        if expected_data_hash and record["data_hash"] != expected_data_hash:
            raise ValueError("RxInfer data_hash does not match trace bundle hash")


def validate_rxinfer_outputs(
    output_dir: Path,
    *,
    expected_data_hash: str | None = None,
) -> list[dict[str, Any]]:
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


def validate_categorical_rxinfer_outputs(
    output_dir: Path,
    *,
    expected_data_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Validate the namespaced categorical RxInfer companion artifacts."""

    free_energy_path = output_dir / "categorical_free_energy_trace.jsonl"
    for filename in (
        "categorical_posteriors.json",
        "categorical_diagnostics.json",
        "categorical_model_config.json",
    ):
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing categorical RxInfer artifact: {path}")
        json.loads(path.read_text(encoding="utf-8"))
    records = load_free_energy_records(free_energy_path)
    validate_free_energy_records(records, expected_data_hash=expected_data_hash)
    if any(
        record.get("model_name") != "categorical_phase_hmm"
        or record.get("mode") != "categorical"
        for record in records
    ):
        raise ValueError(
            "categorical RxInfer records have an unexpected model identity"
        )
    return records


def _cached_rxinfer_outputs(
    trace_bundle_path: Path,
    output_dir: Path,
    *,
    iterations: int,
    categorical: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Return a validated local RxInfer result when its lineage still matches.

    Julia/RxInfer startup and compilation are materially slower than the Python
    producer graph.  The manuscript generator calls this bridge more than once
    during template hydration, so rerunning an identical native inference job
    would make a deterministic render needlessly fragile and exceed the
    template's hydration timeout.  Cache reuse is fail-closed: every artifact
    is revalidated, the trace hash and iteration count must match, and the
    declared model, mode, and pinned toolchain must be unchanged.  Any missing
    or malformed artifact returns ``None`` so the normal native execution path
    repairs it.
    """

    try:
        expected_data_hash = file_sha256(trace_bundle_path)
        if categorical:
            records = validate_categorical_rxinfer_outputs(
                output_dir, expected_data_hash=expected_data_hash
            )
            diagnostics_path = output_dir / "categorical_diagnostics.json"
            expected_model = "categorical_phase_hmm"
            expected_mode = "categorical"
        else:
            records = validate_rxinfer_outputs(
                output_dir, expected_data_hash=expected_data_hash
            )
            diagnostics_path = output_dir / "diagnostics.json"
            expected_model = "gaussian_event_code_model"
            expected_mode = "static"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if not isinstance(diagnostics, dict):
            return None
        if (
            diagnostics.get("data_hash") != expected_data_hash
            or diagnostics.get("iteration_count") != iterations
            or diagnostics.get("model_name") != expected_model
            or diagnostics.get("mode") != expected_mode
            or len(records) != iterations
        ):
            return None
        validate_backend_versions(
            str(diagnostics["julia_version"]),
            str(diagnostics["rxinfer_version"]),
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        RuntimeError,
    ):
        return None
    return records, diagnostics


def _result_from_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    *,
    categorical: bool,
    figure_path: Path,
) -> RxInferRunResult:
    """Build the public bridge result from already validated artifacts."""

    def artifact_path(path: Path) -> str:
        """Serialize project paths and isolated test/output-root paths safely."""

        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    if categorical:
        trace_name = "categorical_free_energy_trace.jsonl"
        posteriors_name = "categorical_posteriors.json"
        diagnostics_name = "categorical_diagnostics.json"
        model_config_name = "categorical_model_config.json"
        figure_name = "rxinfer_categorical_free_energy"
    else:
        trace_name = "free_energy_trace.jsonl"
        posteriors_name = "posteriors.json"
        diagnostics_name = "diagnostics.json"
        model_config_name = "model_config.json"
        figure_name = "rxinfer_free_energy"
    artifact_paths = {
        "free_energy_trace": artifact_path(output_dir / trace_name),
        "posteriors": artifact_path(output_dir / posteriors_name),
        "diagnostics": artifact_path(output_dir / diagnostics_name),
        "model_config": artifact_path(output_dir / model_config_name),
        "stdout": artifact_path(output_dir / "stdout.log"),
        "stderr": artifact_path(output_dir / "stderr.log"),
    }
    return RxInferRunResult(
        run_id=str(diagnostics["run_id"]),
        sample_id=str(diagnostics["sample_id"]),
        model_name=str(diagnostics["model_name"]),
        mode=str(diagnostics["mode"]),
        free_energy_values=tuple(float(record["value"]) for record in records),
        final_free_energy=float(diagnostics["free_energy_final"]),
        data_hash=str(diagnostics["data_hash"]),
        config_hash=str(diagnostics["config_hash"]),
        rxinfer_version=str(diagnostics["rxinfer_version"]),
        julia_version=str(diagnostics["julia_version"]),
        artifact_paths=artifact_paths,
        figure_paths={figure_name: artifact_path(figure_path)},
    )


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
    validate_julia_manifest()
    julia_path = resolve_julia_executable(julia_executable)
    rxinfer_dir = (output_dir or OUTPUT_DIR / "rxinfer").resolve()
    rxinfer_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = (
        OUTPUT_DIR / "figures" if output_dir is None else rxinfer_dir / "figures"
    )
    stdout_path = rxinfer_dir / "stdout.log"
    stderr_path = rxinfer_dir / "stderr.log"

    cached = _cached_rxinfer_outputs(
        trace_bundle_path,
        rxinfer_dir,
        iterations=iterations,
        categorical=False,
    )
    if cached is not None:
        records, diagnostics = cached
        figure_path = plot_rxinfer_free_energy(
            records,
            figure_dir / "rxinfer_free_energy.png",
            title="RxInfer Bethe Free Energy: Gaussian event-code",
        )
        return _result_from_outputs(
            rxinfer_dir,
            records,
            diagnostics,
            categorical=False,
            figure_path=figure_path,
        )

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
    records = validate_rxinfer_outputs(
        rxinfer_dir, expected_data_hash=expected_data_hash
    )
    diagnostics = json.loads(
        (rxinfer_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    validate_backend_versions(
        str(diagnostics["julia_version"]),
        str(diagnostics["rxinfer_version"]),
    )
    figure_path = plot_rxinfer_free_energy(
        records,
        figure_dir / "rxinfer_free_energy.png",
        title="RxInfer Bethe Free Energy: Gaussian event-code",
    )
    return _result_from_outputs(
        rxinfer_dir,
        records,
        diagnostics,
        categorical=False,
        figure_path=figure_path,
    )


def run_rxinfer_categorical_backend(
    trace_bundle_path: Path,
    *,
    output_dir: Path | None = None,
    iterations: int = 8,
    julia_executable: str = "julia",
    instantiate: bool = True,
    model_path: Path | None = None,
) -> RxInferRunResult:
    """Run native categorical inference from shared GNN parameter bytes.

    Gaussian remains an optional numeric-code diagnostic on a separate scale.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    trace_bundle_path = trace_bundle_path.resolve()
    if not trace_bundle_path.exists():
        raise FileNotFoundError(f"missing trace bundle: {trace_bundle_path}")
    validate_julia_manifest()
    julia_path = resolve_julia_executable(julia_executable)
    rxinfer_dir = (output_dir or OUTPUT_DIR / "rxinfer_categorical").resolve()
    rxinfer_dir.mkdir(parents=True, exist_ok=True)
    from .gnn_parser import compile_gnn

    compiled = compile_gnn(model_path or PROJECT_ROOT / "configs/gnn/passive_phase.md")
    shared_model_path = rxinfer_dir / "gnn_model.json"
    shared_model_path.write_text(json.dumps(compiled.to_dict(), allow_nan=False))
    figure_dir = (
        OUTPUT_DIR / "figures" if output_dir is None else rxinfer_dir / "figures"
    )
    stdout_path = rxinfer_dir / "stdout.log"
    stderr_path = rxinfer_dir / "stderr.log"

    cached = _cached_rxinfer_outputs(
        trace_bundle_path,
        rxinfer_dir,
        iterations=iterations,
        categorical=True,
    )
    if cached is not None and cached[1].get("model_digest") == compiled.tensor_digest:
        records, diagnostics = cached
        figure_path = plot_rxinfer_categorical_free_energy(
            records,
            figure_dir / "rxinfer_categorical_free_energy.png",
        )
        return _result_from_outputs(
            rxinfer_dir,
            records,
            diagnostics,
            categorical=True,
            figure_path=figure_path,
        )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    if instantiate:
        instantiated = subprocess.run(
            [
                julia_path,
                f"--project={JULIA_PROJECT}",
                "-e",
                "using Pkg; Pkg.instantiate()",
            ],
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
                "Julia/RxInfer categorical environment instantiation failed with "
                f"exit code {instantiated.returncode}; see {stderr_path}"
            )

    completed = subprocess.run(
        [
            julia_path,
            f"--project={JULIA_PROJECT}",
            str(JULIA_RUNNER),
            "--trace",
            str(trace_bundle_path),
            "--output",
            str(rxinfer_dir),
            "--iterations",
            str(iterations),
            "--model",
            "categorical",
            "--model-spec",
            str(shared_model_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_chunks.append("## run_rxinfer.jl --model categorical\n" + completed.stdout)
    stderr_chunks.append("## run_rxinfer.jl --model categorical\n" + completed.stderr)
    stdout_path.write_text("\n".join(stdout_chunks), encoding="utf-8")
    stderr_path.write_text("\n".join(stderr_chunks), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "Julia/RxInfer categorical backend failed with exit code "
            f"{completed.returncode}; see {stderr_path}"
        )

    expected_data_hash = file_sha256(trace_bundle_path)
    records = validate_categorical_rxinfer_outputs(
        rxinfer_dir, expected_data_hash=expected_data_hash
    )
    diagnostics = json.loads(
        (rxinfer_dir / "categorical_diagnostics.json").read_text(encoding="utf-8")
    )
    validate_backend_versions(
        str(diagnostics["julia_version"]),
        str(diagnostics["rxinfer_version"]),
    )
    figure_path = plot_rxinfer_categorical_free_energy(
        records,
        figure_dir / "rxinfer_categorical_free_energy.png",
    )
    return _result_from_outputs(
        rxinfer_dir,
        records,
        diagnostics,
        categorical=True,
        figure_path=figure_path,
    )


def run_shared_gnn_native(
    model, observations: list[int], output_dir: Path
) -> dict[str, Any]:
    """Execute native smoothing from shared validated bytes, without stale-cache reuse."""
    import numpy as np

    from .gnn_experiments import envelope, model_payload, source_hashes, write_payload
    from .gnn_model import CompiledModel
    from .gnn_reference import forward_backward

    model = CompiledModel.from_dict(model.to_dict())
    reference = forward_backward(model, observations)
    validate_julia_manifest()
    julia = resolve_julia_executable()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "model.json"
    obs_path = output_dir / "observations.json"
    result_path = output_dir / "result.json"
    write_payload(spec_path, model_payload(model))
    write_payload(
        obs_path,
        {
            "observations": observations,
            "model_digest": model.tensor_digest,
            "model_file_sha256": file_sha256(spec_path),
            "tensor_contract_json": json.dumps(
                {
                    k: v
                    for k, v in model.to_dict().items()
                    if k not in ("source_sha256", "annotations", "name", "model_digest")
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        },
    )
    code = "using ActiveAssemblyRxInfer; ActiveAssemblyRxInfer.run_shared_categorical(ARGS[1],ARGS[2],ARGS[3])"
    previous_mtime = result_path.stat().st_mtime_ns if result_path.exists() else None
    started = time.time_ns()
    try:
        completed = subprocess.run(
            [
                julia,
                "--startup-file=no",
                f"--project={JULIA_PROJECT}",
                "-e",
                code,
                str(spec_path),
                str(obs_path),
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        )
        (output_dir / "stdout.log").write_text(stdout)
        (output_dir / "stderr.log").write_text(stderr)
        raise RuntimeError(
            f"GNN_NATIVE_TIMEOUT; see {output_dir / 'stderr.log'}"
        ) from exc
    (output_dir / "stdout.log").write_text(completed.stdout)
    (output_dir / "stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"GNN_NATIVE_FAILED exit={completed.returncode}; see {output_dir / 'stderr.log'}"
        )
    if (
        not result_path.is_file()
        or result_path.stat().st_mtime_ns < started
        or result_path.stat().st_mtime_ns == previous_mtime
    ):
        raise RuntimeError("GNN_NATIVE_OUTPUT_NOT_FRESH")
    result = json.loads(result_path.read_text())
    validate_backend_versions(result["julia_version"], result["rxinfer_version"])
    if (
        result["model_digest"] != model.tensor_digest
        or result["model_file_sha256"] != file_sha256(spec_path)
        or result["trace_sha256"] != file_sha256(obs_path)
    ):
        raise ValueError("GNN_NATIVE_LINEAGE_MISMATCH")
    native = np.array(result["native_smoothed"], dtype=float)
    expected = np.array(reference["smoothed"], dtype=float)
    if (
        native.shape != expected.shape
        or not np.isfinite(native).all()
        or (native < 0).any()
        or not np.allclose(native.sum(axis=1), 1, atol=1e-5)
    ):
        raise RuntimeError("GNN_NATIVE_INVALID_POSTERIOR")
    error = float(np.max(np.abs(native - expected)))
    if error > 1e-5:
        raise RuntimeError(f"GNN_NATIVE_PARITY_FAILED {error}")
    return envelope(
        {
            **result,
            "python_smoothed": reference["smoothed"],
            "observations": observations,
            "max_abs_error": error,
            "tolerance": 1e-5,
            "input_hashes": {
                **source_hashes(),
                **{
                    str(p.relative_to(PROJECT_ROOT)): file_sha256(p)
                    for p in (
                        Path(__file__),
                        JULIA_PROJECT / "src/ActiveAssemblyRxInfer.jl",
                        JULIA_PROJECT / "Project.toml",
                        JULIA_MANIFEST,
                    )
                },
            },
            "native_source_sha256": file_sha256(
                JULIA_PROJECT / "src/ActiveAssemblyRxInfer.jl"
            ),
        }
    )
